#!/usr/bin/env python3

import logging
import os
import sys
import traceback
import glob
import cv2,threading
import numpy as np
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
from natsort import natsorted
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import psutil
import time 

# OpenMP ve threading çakışmalarını önle
os.environ.update({
    'KMP_DUPLICATE_LIB_OK': 'TRUE',
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'KMP_WARNINGS': '0'
})

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA: {torch.version.cuda}, Devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

print(f"NumPy: {np.__version__}, OpenCV: {cv2.__version__}")

# PyQt5 imports
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUiType

# Custom modules
from saicinpainting.training.trainers import load_checkpoint
from maskCreator import PolygonDrawer

# Performans ayarları
NUM_WORKERS = min(4, psutil.cpu_count())

def resource_path(relative_path):
    """Get resource path for PyInstaller compatibility"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Load UI
ui_path = resource_path('panoComplete.ui')
ui, _ = loadUiType(ui_path)

class QTextEditLogger(QObject, logging.Handler):
    log_signal = pyqtSignal(str, int)
    
    def __init__(self, parent):
        QObject.__init__(self)
        logging.Handler.__init__(self)
        self.widget = parent.log
        self.widget.setReadOnly(True)
        
        self.colors = {
            logging.DEBUG: QColor("#3b82f6"),
            logging.INFO: QColor("#1e293b"),
            logging.WARNING: QColor("#f59e0b"),
            logging.ERROR: QColor("#ef4444"),
            logging.CRITICAL: QColor("#b91c1c")
        }
        
        # Filter model structure keywords
        self.filtered_keywords = [
            "Conv2d", "BatchNorm2d", "ReLU", "FFC", "SpectralTransform", 
            "FourierUnit", "Sequential", "Identity", "ConcatTupleLayer",
            "ConvTranspose2d", "ReflectionPad2d", "Sigmoid", "FFCResnetBlock"
        ]
        
        self.log_signal.connect(self.write_log)

    def emit(self, record):
        msg = self.format(record)
        if not any(keyword in msg for keyword in self.filtered_keywords):
            self.log_signal.emit(msg, record.levelno)
    
    @pyqtSlot(str, int)
    def write_log(self, msg, level):
        log_format = QTextCharFormat()
        log_format.setForeground(self.colors.get(level, QColor("#1e293b")))
        
        font = log_format.font()
        font.setPointSize(10)
        log_format.setFont(font)
    
        cursor = self.widget.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(msg, log_format)
        self.widget.setTextCursor(cursor)
        self.widget.ensureCursorVisible()

class ProcessingWorker(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()
    log_signal = pyqtSignal(str, int)
    error_signal = pyqtSignal(str)

    def __init__(self, model, predict_config, image_files, out_paths, custom_mask=None):
        super().__init__()
        self.model = model
        self.predict_config = predict_config
        self.image_files = image_files
        self.out_paths = out_paths
        self.is_running = True
        self.custom_mask = custom_mask

        # Dosya yazma için thread kilidi ekleyin
        self.file_lock = threading.Lock()
      
        
        # ✨ PERFORMANS OPTİMİZASYONU
        self.device = next(self.model.parameters()).device
        
        # Batch processing için buffer
        self.batch_size = 2 if torch.cuda.is_available() else 1
        self.prefetch_buffer = 3
        
    def stop(self):
        self.is_running = False
        
    def run(self):
        try:
            # ✨ Optimize edilmiş mask hazırlama
            masks = self._prepare_masks_optimized()
            if not masks or len(masks) != 3:
                self.log_signal.emit("Mask hazırlama hatası!", logging.ERROR)
                return
                
            total_files = len(self.image_files)
            if total_files == 0:
                self.log_signal.emit("İşlenecek resim bulunamadı!", logging.ERROR)
                return
                
            self.log_signal.emit(f"Toplam {total_files} resim işlenecek (Batch size: {self.batch_size})", logging.INFO)
            
            # ✨ GPU bellek optimizasyonu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
                
            # ✨ Batch processing ile hızlandırılmış işlem
            self._process_images_batch(total_files, masks)
            self.log_signal.emit("İşlem tamamlandı!", logging.INFO)
            self.finished_signal.emit()
            
        except Exception as e:
            self.log_signal.emit(f"İşlem hatası: {str(e)}", logging.ERROR)
            self.error_signal.emit(f"İşlem hatası: {str(e)}")
            self.finished_signal.emit()
        finally:
            # ✨ Bellek temizliği
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _prepare_masks_optimized(self):
        """Optimize edilmiş mask hazırlama - Akıllı boyut kontrolü"""
        
        # Cache için ilk görüntüyü yükle
        if not hasattr(self, '_cached_img_size'):
            first_img = cv2.imread(self.image_files[0])
            if first_img is None:
                raise Exception("İlk görüntü yüklenemedi!")
            
            self._cached_img_size = {
                'orig_width': first_img.shape[1],
                'orig_height': first_img.shape[0],
                'target_width': int(first_img.shape[1] / 10),
                'target_height': int(first_img.shape[0] / 10)
            }
            del first_img
        
        size = self._cached_img_size
        
        # ✅ CUSTOM MASK KONTROLÜ
        if self.custom_mask is not None:
            self.log_signal.emit("Özel mask'tan 3 mask oluşturuluyor...", logging.INFO)
            
            # ✨ AKILLI BOYUT KONTROLÜ
            mask_height, mask_width = self.custom_mask.shape
            orig_width, orig_height = size['orig_width'], size['orig_height']
            target_width, target_height = size['target_width'], size['target_height']
            
            masks = {}
            
            # Ana mask - Boyut kontrolü ile optimize resize
            if (mask_width == orig_width and mask_height == orig_height):
                # ✅ Zaten doğru boyutta, resize gereksiz!
                masks['main'] = self.custom_mask.copy()
                self.log_signal.emit("✅ Mask ile fotoğraflar aynı boyutta", logging.INFO)
            else:
                # ❌ BOYUT UYUMSUZLUĞU - HATA VER
                error_msg = f"❌ Mask boyutu resimlerle uyumlu değil!\n\n"
                error_msg += f"📏 Resim boyutu: {orig_width} x {orig_height}\n"
                error_msg += f"📐 Mask boyutu: {mask_width} x {mask_height}\n\n"
                error_msg += "🔧 Çözüm seçenekleri:\n"
                error_msg += "1. 'Create Mask' ile doğru boyutta yeni mask oluşturun\n"
                error_msg += "2. Mevcut resimlerle aynı boyutta farklı bir mask seçin\n"
                error_msg += "3. Mask'ı harici bir araçla doğru boyuta getirin"
                
                self.log_signal.emit(error_msg, logging.ERROR)
                raise Exception(f"Mask boyut uyumsuzluğu: Resim {orig_width}x{orig_height}, Mask {mask_width}x{mask_height}")

            # Küçük mask - Her zaman resize gerekli (1/10 boyut)
            masks['small'] = cv2.resize(self.custom_mask, (target_width, target_height), cv2.INTER_AREA)
            
            # Ters mask
            masks['inverted'] = cv2.bitwise_not(masks['main'])
            
            self.log_signal.emit("✅ 3 mask başarıyla oluşturuldu", logging.INFO)
            return masks
        
        # ❌ MASK YOKSA HATA
        error_msg = "❌ Mask bulunamadı!\n\n"
        error_msg += "İşleme devam etmek için:\n"
        error_msg += "1. 'Create Mask' ile yeni mask oluşturun, VEYA\n"
        error_msg += "2. 'Select Mask' ile mevcut mask dosyası seçin"
        
        self.log_signal.emit(error_msg, logging.ERROR)
        raise Exception("Mask gerekli! Lütfen önce mask oluşturun veya seçin.")


    
    def _process_images_batch(self, total_files, masks):
        """Batch processing ile hızlandırılmış görüntü işleme"""
        processed_count = 0
        
        # ✨ İşlenen resimler dosyası oluştur
        processed_log_file = os.path.join(self.out_paths, "işlenen_resimler.txt")
        os.makedirs(os.path.dirname(processed_log_file), exist_ok=True)
        self.processed_files = []
        if os.path.exists(processed_log_file):
            with open(processed_log_file, 'r', encoding='utf-8') as f:
                self.processed_files = [line.strip() for line in f.readlines()]

        mask_main = masks['main']
        mask_small = masks['small'] 
        mask_inverted = masks['inverted']
        
        # Mask'ları önceden normalize et
        mask_main_norm = mask_main.astype(np.float32) / 255.0
        mask_inverted_norm = mask_inverted.astype(np.float32) / 255.0
        
        if len(mask_main_norm.shape) == 2:
            mask_main_3ch = np.stack([mask_main_norm, mask_main_norm, mask_main_norm], axis=2)
        else:
            mask_main_3ch = mask_main_norm
            
        if len(mask_inverted_norm.shape) == 2:
            mask_inverted_3ch = np.stack([mask_inverted_norm, mask_inverted_norm, mask_inverted_norm], axis=2)
        else:
            mask_inverted_3ch = mask_inverted_norm

        images = self.image_files.copy()
        for img_path in images:
            ## skip already processed files
            img_name = os.path.basename(img_path)
            directory = os.path.basename(os.path.dirname(img_path))
            if f"{directory}-{img_name}" in self.processed_files:
                processed_count += 1
                self.image_files.remove(img_path)
                

        self.log_signal.emit(f"skiping {processed_count} files processed", logging.INFO)

        # ✨ BATCH PROCESSING
        for i in range(0, len(self.image_files), self.batch_size):
            if not self.is_running:
                break
                
            batch_files = self.image_files[i:i + self.batch_size]
            batch_results = []
            
            try:
                # Batch için görüntüleri paralel yükle
                with ThreadPoolExecutor(max_workers=min(len(batch_files), 4)) as executor:
                    load_futures = {executor.submit(self._load_image_fast, img_path): img_path 
                                for img_path in batch_files}
                    
                    loaded_images = {}
                    for future in load_futures:
                        img_path = load_futures[future]
                        try:
                            loaded_images[img_path] = future.result()
                        except Exception as e:
                            self.log_signal.emit(f"Yükleme hatası {os.path.basename(img_path)}: {str(e)}", logging.ERROR)
                            continue
                
                # Model inference batch olarak
                if loaded_images:
                    batch_data = []
                    batch_orig_imgs = []
                    batch_paths = []
                    
                    for img_path, (downed_img, roi_img, orig_img) in loaded_images.items():
                        data = self._prepare_data_fast(downed_img, mask_small, roi_img)
                        batch_data.append(data)
                        batch_orig_imgs.append(orig_img)
                        batch_paths.append(img_path)
                    
                    # ✨ BATCH INFERENCE
                    if batch_data:
                        batch_results = self._run_batch_inference(batch_data)
                        
                        # Post-processing paralel olarak
                        with ThreadPoolExecutor(max_workers=min(len(batch_results), 4)) as executor:
                            save_futures = []
                            
                            for j, (result, orig_img, img_path) in enumerate(zip(batch_results, batch_orig_imgs, batch_paths)):
                                future = executor.submit(
                                    self._postprocess_and_save,
                                    result, orig_img, img_path,
                                    mask_main_3ch, mask_inverted_3ch
                                )
                                save_futures.append(future)
                            
                            # ✨ Sonuçları topla ve dosya adlarını logla
                            for j, (future, img_path) in enumerate(zip(save_futures, batch_paths)):
                                try:
                                    if future.result():
                                        processed_count += 1
                                        img_name = os.path.basename(img_path)
                                        directory = os.path.basename(os.path.dirname(img_path))

                                        # ✨ Log'a resim adı ile birlikte yaz
                                        self.log_signal.emit(f"İşlendi: {img_name} ({processed_count}/{total_files})", logging.INFO)
                                        
                                        # ✨ İşlenen resimler dosyasına anlık kaydet
                                        try:
                                            with self.file_lock: 
                                                with open(processed_log_file, 'a', encoding='utf-8') as f:
                                                    f.write(f"{directory}-{img_name}\n")
                                                    f.flush()  # Anlık yazma için

                                        except Exception as log_error:
                                            self.log_signal.emit(f"Log yazma hatası: {str(log_error)}", logging.WARNING)
                                            
                                except Exception as e:
                                    img_name = os.path.basename(batch_paths[j])
                                    self.log_signal.emit(f"Kaydetme hatası - {img_name}: {str(e)}", logging.ERROR)
                                    
                                    # ✨ Hatalı dosyaları da logla
                                    try:
                                        with open(processed_log_file, 'a', encoding='utf-8') as f:
                                            timestamp = time.strftime('%H:%M:%S')
                                            f.write(f"[{timestamp}] HATA: {img_name} - {str(e)}\n")
                                            f.flush()
                                    except:
                                        pass
                
                # Progress güncelle
                progress = int(((i + len(batch_files)) / total_files) * 100)
                self.progress_signal.emit(progress)
                
            except Exception as e:
                self.log_signal.emit(f"Batch işleme hatası: {str(e)}", logging.ERROR)
                continue
        
        
        self.log_signal.emit(f"Toplam {processed_count}/{total_files} resim başarıyla işlendi", logging.INFO)
        if processed_count < total_files:
            self.log_signal.emit(f"{total_files - processed_count} resim işlenemedi", logging.WARNING)
    
    def _load_image_fast(self, img_path):
        """Hızlandırılmış görüntü yükleme"""
        # ✨ Bellek etkili yükleme
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception(f"Resim yüklenemedi: {img_path}")
        
        size = self._cached_img_size
        orig_img = img  # Copy yerine direkt kullan
        
        # ROI işlemi basitleştirildi
        roi_img = img
        
        # ✨ Resize optimizasyonu
        downed_img = cv2.resize(img, (size['target_width'], size['target_height']), interpolation=cv2.INTER_AREA)
        downed_img = np.transpose(downed_img, (2, 0, 1)).astype('float32') / 255.0
        
        return downed_img, roi_img, orig_img
    
    def _prepare_data_fast(self, downed_img, mask_small, roi_img):
        """Hızlandırılmış veri hazırlama"""
        data = {
            "image": self._pad_to_modulo(downed_img, 8),
            "mask": self._pad_to_modulo(np.expand_dims(mask_small, axis=0), 8),
            "unpad_to_size": list(roi_img.shape[:2])
        }
        
        return {
            'image': torch.from_numpy(data['image']).to(self.device, non_blocking=True),
            'mask': torch.from_numpy(data['mask']).to(self.device, non_blocking=True),
            'unpad_to_size': data['unpad_to_size']
        }
    
    def _run_batch_inference(self, batch_data_list):
        """Batch inference ile hızlandırma"""
        results = []
        with torch.no_grad():
            for data in batch_data_list:
                batch = default_collate([data])
                batch['mask'] = (batch['mask'] > 0).float()
                batch = self.model(batch)
                batch['inpainted'] = batch['inpainted'] * batch['mask'] + batch['image'] * (1 - batch['mask'])
                
                result = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
                h, w = data['unpad_to_size']
                results.append(result[:h, :w, :])
        
        return results
    
    def _postprocess_and_save(self, result, orig_img, img_path, mask_main_3ch, mask_inverted_3ch):
        """Hızlandırılmış post-processing ve kaydetme"""
        try:
            imgName = os.path.basename(img_path)
            dirname = os.path.basename(os.path.dirname(img_path))
            cur_out_fname = os.path.join(self.out_paths, dirname, imgName)
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            
            # ✨ Optimize edilmiş normalizasyon
            if len(result.shape) == 3 and result.shape[2] == 3:
                result_normalized = np.clip(result * 255, 0, 255).astype('uint8')
            else:
                result_single = np.clip(result * 255, 0, 255).astype('uint8')
                if len(result_single.shape) == 2:
                    result_normalized = cv2.cvtColor(result_single, cv2.COLOR_GRAY2BGR)
                else:
                    result_normalized = result_single
            
            # ✨ Tek seferde resize ve işlem
            result_resized = cv2.resize(result_normalized, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            # ✨ Vectorized operations
            orig_img_norm = orig_img.astype(np.float32) / 255.0
            result_resized_norm = result_resized.astype(np.float32) / 255.0
            
            # Manuel maskeleme (vectorized)
            inpainted_part = result_resized_norm * mask_main_3ch
            preserved_part = orig_img_norm * mask_inverted_3ch
            final_image_norm = inpainted_part + preserved_part
            
            # Tek seferde clip ve convert
            final_image = np.clip(final_image_norm * 255, 0, 255).astype(np.uint8)
            
            # ✨ Optimize edilmiş kaydetme
            success = cv2.imwrite(cur_out_fname, final_image, [
                int(cv2.IMWRITE_JPEG_QUALITY), 90
            ])
            
            return success
            
        except Exception as e:
            self.log_signal.emit(f"Post-processing hatası {imgName}: {str(e)}", logging.ERROR)
            return False
    
    def _pad_to_modulo(self, img, modulo):
        """Pad image to modulo"""
        if len(img.shape) == 3:
            channels, height, width = img.shape
            padded_height = (height + modulo - 1) // modulo * modulo
            padded_width = (width + modulo - 1) // modulo * modulo
            return np.pad(img, ((0, 0), (0, padded_height - height), (0, padded_width - width)), mode='reflect')
        else:
            height, width = img.shape
            padded_height = (height + modulo - 1) // modulo * modulo
            padded_width = (width + modulo - 1) // modulo * modulo
            return np.pad(img, ((0, padded_height - height), (0, padded_width - width)), mode='reflect')

class ModelLoaderThread(QThread):
    finished_signal = pyqtSignal(bool, object, object)
    progress_signal = pyqtSignal(str)
    
    def run(self):
        try:
            self.progress_signal.emit("Model yükleniyor...")
            
            # Load configs
            config_path = resource_path("./configs/prediction/default.yaml")
            with open(config_path, 'r') as f:
                predict_config = OmegaConf.create(yaml.safe_load(f))
            
            train_config_path = resource_path("big-lama/config.yaml")
            with open(train_config_path, 'r') as f:
                train_config = OmegaConf.create(yaml.safe_load(f))

            train_config.training_model.predict_only = True
            train_config.visualizer.kind = 'noop'

            # Load model
            checkpoint_path = resource_path(f"big-lama/models/{predict_config.model.checkpoint}")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Suppress model output
            original_repr = torch.nn.Module.__repr__
            torch.nn.Module.__repr__ = lambda self: f"{self.__class__.__name__} (suppressed)"
            
            try:
                model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=device)
                model.to(device).freeze()
                self.finished_signal.emit(True, (model, predict_config), None)
            finally:
                torch.nn.Module.__repr__ = original_repr
                
        except Exception as e:
            self.finished_signal.emit(False, None, f"Model yükleme hatası: {str(e)}")

class MainApp(QMainWindow, ui):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        
        # Initialize variables
        self.imgPaths = ""
        self.outPaths = ""
        self.custom_mask = None
        self.processing_thread = None
        self.model = None
        self.predict_config = None
        self.processed_files_set = None
        
        self._setup_ui()
        self._setup_logging()
        self._load_model()
        
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _setup_ui(self):
        """Setup UI with modern theme"""
        self.setWindowTitle("ImageModifier")
        self.setWindowIcon(QIcon(resource_path("icon.ico")))
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e293b; color: white; }
            QGroupBox {
                font-size: 14px; font-weight: bold; color: white;
                border: 2px solid #64748b; border-radius: 10px;
                margin-top: 15px; padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top center;
                padding: 0 10px; background-color: #334155;
            }
            QLineEdit {
                background-color: #334155; color: white; border: 1px solid #64748b;
                border-radius: 8px; padding: 8px; font-size: 13px; min-height: 20px;
            }
            QLineEdit:focus { border: 2px solid #93c5fd; }
            QPushButton {
                border: none; border-radius: 8px; padding: 10px 20px;
                font-weight: bold; font-size: 13px;
            }
            QPushButton:disabled { background-color: #94a3b8; color: #e2e8f0; }
            QProgressBar {
                border: 1px solid #64748b; border-radius: 8px; text-align: center;
                background-color: #334155; color: white; font-weight: bold;
            }
            QProgressBar::chunk { background-color: #60a5fa; border-radius: 7px; }
        """)
        
        # Create layout
        self._create_layout()
        self._connect_buttons()
        
        # Initial state
        self.baslat.setEnabled(False)
        self.durdur.setEnabled(False)
        self.durum.setText("Model yükleniyor...")
        
        self.resize(1250, 600)
        self._center_window()

    def _create_layout(self):
        """Create main layout"""
        main_container = QWidget()
        main_layout = QHBoxLayout(main_container)
        
        # Left panel
        left_panel = self._create_left_panel()
        # Right panel  
        right_panel = self._create_right_panel()
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 2)
        
        self.setCentralWidget(main_container)

    def _create_left_panel(self):
        """Create left control panel"""
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Title
        title = QLabel("ImageModifier")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 24px; font-weight: bold; color: white; padding: 15px;
            background-color: #334155; border-radius: 10px; margin-bottom: 10px;
        """)
        left_layout.addWidget(title)
        
        # Input group
        input_group = QGroupBox("Ayarlar")
        input_layout = QGridLayout(input_group)
        
        # Image folder
        input_layout.addWidget(QLabel("Image Folder:"), 0, 0)
        self.inDir.setPlaceholderText("Select image folder...")
        input_layout.addWidget(self.inDir, 0, 1)
        
        self.inDirBtn = QPushButton("Browse...")
        self._style_button(self.inDirBtn, "#60a5fa", "#3b82f6")
        input_layout.addWidget(self.inDirBtn, 0, 2)
        
        # Output folder
        input_layout.addWidget(QLabel("Output Folder:"), 1, 0)
        self.outDir.setPlaceholderText("Select output folder...")
        input_layout.addWidget(self.outDir, 1, 1)
        
        self.outDirBtn = QPushButton("Browse...")
        self._style_button(self.outDirBtn, "#60a5fa", "#3b82f6")
        input_layout.addWidget(self.outDirBtn, 1, 2)
        
        # Mask buttons
        mask_container = QWidget()
        mask_layout = QHBoxLayout(mask_container)
        
        self.createMaskBtn = QPushButton("Maske Oluştur")
        self._style_button(self.createMaskBtn, "#8b5cf6", "#7c3aed")
        self.selectMaskBtn = QPushButton("Maske Seç")
        self._style_button(self.selectMaskBtn, "#60a5fa", "#3b82f6")
        
        mask_layout.addWidget(self.createMaskBtn)
        mask_layout.addWidget(self.selectMaskBtn)
        input_layout.addWidget(mask_container, 2, 0, 1, 3)
        
        left_layout.addWidget(input_group)
        
        # Control group
        control_group = QGroupBox("Proses Kontrolü")
        control_layout = QVBoxLayout(control_group)
        
        # Buttons
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        
        self._style_button(self.baslat, "#34d399", "#10b981")
        self._style_button(self.durdur, "#f87171", "#ef4444")
        self.baslat.setText("Proses Başlat")
        self.durdur.setText("Proses Durdur")
        
        button_layout.addWidget(self.baslat)
        button_layout.addWidget(self.durdur)
        control_layout.addWidget(button_container)
        
        # Status and progress
        self.durum.setAlignment(Qt.AlignCenter)
        self.durum.setStyleSheet("""
            color: #34d399; font-weight: bold; font-size: 16px;
            background-color: #334155; border-radius: 5px; padding: 8px;
        """)
        control_layout.addWidget(self.durum)
        
        self.progressBar = QProgressBar()
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(True)
        self.progressBar.setAlignment(Qt.AlignCenter)
        self.progressBar.setFormat("%p% Completed")
        control_layout.addWidget(self.progressBar)
        
        left_layout.addWidget(control_group)
        left_layout.addStretch()
        
        return left_panel

    def _create_right_panel(self):
        """Create right log panel"""
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        log_group = QGroupBox("Operation Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log.setStyleSheet("""
            background-color: white; color: #1e293b; border: 1px solid #64748b;
            border-radius: 5px; padding: 10px; font-family: Consolas, Monaco, monospace;
            font-size: 14px;
        """)
        self.log.setMinimumHeight(400)
        log_layout.addWidget(self.log)
        
        right_layout.addWidget(log_group)
        return right_panel

    def _style_button(self, button, bg_color, hover_color):
        """Apply button styling"""
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color}; color: white; min-height: 40px;
            }}
            QPushButton:hover {{ background-color: {hover_color}; }}
        """)

    def _connect_buttons(self):
        """Connect button signals"""
        self.inDirBtn.clicked.connect(lambda: self._select_directory(True))
        self.outDirBtn.clicked.connect(lambda: self._select_directory(False))
        self.baslat.clicked.connect(self._start_processing)
        self.durdur.clicked.connect(self._stop_processing)
        self.createMaskBtn.clicked.connect(self._create_mask)
        self.selectMaskBtn.clicked.connect(self._select_mask)

    def _setup_logging(self):
        """Setup logging system"""
        logTextBox = QTextEditLogger(self)
        logTextBox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s\n'))
        logging.getLogger().addHandler(logTextBox)
        logging.getLogger().setLevel(logging.INFO)

    def _load_model(self):
        """Load model in background"""
        self.model_loader = ModelLoaderThread()
        self.model_loader.progress_signal.connect(lambda msg: self.durum.setText(msg))
        self.model_loader.finished_signal.connect(self._on_model_loaded)
        self.model_loader.start()

    def _on_model_loaded(self, success, result, error_message):
        """Handle model loading result"""
        if success:
            self.model, self.predict_config = result
            logging.info("Model başarıyla yüklendi!")
            self.durum.setText("Ready")
            self.baslat.setEnabled(True)
        else:
            logging.error(error_message)
            self.durum.setText("Model yüklenemedi!")
            QMessageBox.critical(self, "Model Hatası", error_message)

    def _select_directory(self, is_input):
        """Select input or output directory"""
        caption = "Resim klasörünü seçin" if is_input else "Çıktı klasörünü seçin"
        directory = QFileDialog.getExistingDirectory(self, caption)
        
        if directory:
            if is_input:
                self.inDir.setText(directory)
                self.imgPaths = directory
                logging.info(f"Resim klasörü: {directory}")
            else:
                self.outDir.setText(directory)
                self.outPaths = directory
                logging.info(f"Çıktı klasörü: {directory}")

    def _start_processing(self):
        """Start image processing"""
        if not self.imgPaths or not self.outPaths:
            QMessageBox.warning(self, "Hata", "Lütfen klasörleri seçin!")
            return
        
        # Find image files
        formats = ["*.jpeg", "*.jpg", "*.png", "*.bmp", "*.tiff", "*.tif"]
        image_files = []
        
        for fmt in formats:
            image_files.extend(glob.glob(os.path.join(self.imgPaths, "*", fmt)))
            image_files.extend(glob.glob(os.path.join(self.imgPaths, fmt)))
            
        if not image_files:
            QMessageBox.warning(self, "Hata", "İşlenecek resim bulunamadı!")
            return
        
        image_files = natsorted(image_files)
        
       
        
            
        
        logging.info(f"{len(image_files)} resim işlenecek")
        if len(image_files) == 0:
            QMessageBox.information(self, "Bilgi", "İşlenecek yeni resim bulunmadı, tüm resimler zaten işlenmiş!")
            return
        
        # Update UI
        self.baslat.setEnabled(False)
        self.durdur.setEnabled(True)
        self.durum.setText("İşleniyor...")
        self.progressBar.setValue(0)
        
        # Start processing thread
        self.processing_thread = ProcessingWorker(
            self.model, self.predict_config, image_files, self.outPaths, self.custom_mask
        )

        self.processing_thread.progress_signal.connect(self._update_progress)
        self.processing_thread.finished_signal.connect(self._process_finished)
        self.processing_thread.log_signal.connect(lambda msg, level: logging.log(level, msg))
        self.processing_thread.error_signal.connect(lambda msg: QMessageBox.critical(self, "Hata", msg))
        
        self.processing_thread.start()

    def _update_progress(self, value):
        """Update progress bar"""
        self.progressBar.setValue(value)
        self.durum.setText(f"İşleniyor... %{value}")

    def _process_finished(self):
        """Handle process completion"""
        self.baslat.setEnabled(True)
        self.durdur.setEnabled(False)
        self.durum.setText("Tamamlandı!")
        self.progressBar.setValue(100)
        QMessageBox.information(self, "Başarılı", "İşlem tamamlandı!")

    def _stop_processing(self):
        """Stop processing"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait()
            self.baslat.setEnabled(True)
            self.durdur.setEnabled(False)
            self.durum.setText("Durduruldu!")
            self.progressBar.setValue(0)

    def _create_mask(self):
        """Create custom mask"""
        if not self.imgPaths:
            QMessageBox.warning(self, "Hata", "Önce resim klasörünü seçin!")
            return
            
        # Find first image
        formats = ["*.jpeg", "*.jpg", "*.png", "*.bmp", "*.tiff", "*.tif"]
        image_files = []
        
        for fmt in formats:
            image_files.extend(glob.glob(os.path.join(self.imgPaths, "*", fmt)))
            image_files.extend(glob.glob(os.path.join(self.imgPaths, fmt)))
            
        if not image_files:
            QMessageBox.warning(self, "Hata", "Resim bulunamadı!")
            return
            
        first_image = natsorted(image_files)[0]
        img = cv2.imread(first_image)
        
        if img is None:
            QMessageBox.warning(self, "Hata", "Resim yüklenemedi!")
            return
            
        QMessageBox.information(self, "Mask Creation", 
            "Controls:\n" +
            "• Left click: Add point\n" +
            "• 'f': Complete polygon\n" +
            "• 'c': Clear\n" +
            "• 'q': Save and exit\n" +
            "• ESC: Cancel"
        )
        
       
        
        polygon_drawer = PolygonDrawer(img, output_path=None)
        result = polygon_drawer.start_drawing()
        
        if result:
            # Kullanıcının seçtiği dizini polygon_drawer.output_path'den al
            mask_path = os.path.join(polygon_drawer.output_path, "mask_main.png")
            logging.info(f"Mask kaydedildi: {mask_path}")
            if os.path.exists(mask_path):
                self.custom_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                logging.info(f"Özel mask oluşturuldu: {mask_path}")
                QMessageBox.information(self, "Başarılı", f"Mask oluşturuldu: {mask_path}")
            else:
                QMessageBox.warning(self, "Hata", f"Mask dosyası bulunamadı: {mask_path}")
        else:
            QMessageBox.warning(self, "İptal", "Mask oluşturma iptal edildi")

    def _select_mask(self):
        """Select existing mask file"""
        if not self.imgPaths:
            QMessageBox.warning(self, "Hata", "Önce resim klasörünü seçin!")
            return
        mask_file, _ = QFileDialog.getOpenFileName(
            self, "Mask Dosyası Seç", "mask",
            "Resim Dosyaları (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if mask_file:
            self.custom_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if self.custom_mask is not None:
                logging.info(f"Mask yüklendi: {mask_file}")
                QMessageBox.information(self, "Başarılı", "Mask yüklendi!")
                height, width = self.custom_mask.shape
                unique_values = np.unique(self.custom_mask)
                logging.info(f"Mask yüklendi: {mask_file}")
                logging.info(f"Mask Boyutu: {width}x{height}")
                logging.info(f"Değer aralığı: {unique_values.min()}-{unique_values.max()}")
            
            else:
                QMessageBox.warning(self, "Hata", "Mask yüklenemedi!")

    def _center_window(self):
        """Center window on screen"""
        frame_geometry = self.frameGeometry()
        screen_center = QDesktopWidget().availableGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Show splash screen
    splash_pix = QPixmap(400, 200)
    splash_pix.fill(QColor("#1e293b"))
    
    painter = QPainter(splash_pix)
    painter.setPen(QColor("#ffffff"))
    painter.setFont(QFont("Arial", 14, QFont.Bold))
    painter.drawText(QRect(0, 0, 400, 200), Qt.AlignCenter, 
                    "ImageModifier\nLoading...")
    painter.end()
    
    splash = QSplashScreen(splash_pix)
    splash.show()
    app.processEvents()
    
    try:
        window = MainApp()
        window.show()
        splash.finish(window)
        return app.exec_()
    except Exception as e:
        QMessageBox.critical(None, "Kritik Hata", f"Uygulama başlatılamadı:\n{str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())