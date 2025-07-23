import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QApplication
import os
import sys

class PolygonDrawer:
    def __init__(self, img, output_path=None):
        self.img = img
        self.output_path = output_path  # Varsayılan olarak None olabilir
        self.current_pts = []
        self.polygons = []
        self.drawing = False
        self.load_image()
        
    def load_image(self):
        """Resmi yükle ve görüntüleme için hazırla"""
        if self.img is None:
            raise Exception("Image not found. Make sure the path is correct.")

        # Görüntüleme için resmi yeniden boyutlandır
        self.window_width = 1328
        self.window_height = 714
        height, width = self.img.shape[:2]
        aspect_ratio = width / height

        # En-boy oranını koru
        if aspect_ratio > self.window_width / self.window_height:
            new_width = self.window_width
            new_height = int(self.window_width / aspect_ratio)
        else:
            new_height = self.window_height
            new_width = int(self.window_height * aspect_ratio)

        # Daha verimli yeniden boyutlandırma için INTER_AREA kullan
        self.img_display = cv2.resize(self.img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        self.img_display_original = self.img_display.copy()
        self.new_width = new_width
        self.new_height = new_height
        self.height = height
        self.width = width
        
    def draw_info_panel(self, img):
        """Bilgi paneli çiz"""
        # Panel arka planı
        panel_height = 210
        panel_width = 480
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (47, 79, 79), -1)  # Koyu gri arka plan
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)  # Yarı saydam efekt
        
        # Panel kenarlığı
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 255, 0), 2)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
        
        # Panel başlığı ve içeriği
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_font_scale = 0.9
        font_scale = 0.7
        line_height = 30
        text_color = (255, 255, 255)  # Beyaz metin
        
        # Panel içeriği
        texts = [
            "MASK CREATION CONTROLS",
            "Left Click: Add Point",
            "'f' key: Complete Polygon",
            "'c' key: Clear ",
            "'q' key: Finish and Create Mask",
            "ESC key: Exit"
        ]
        
        # Metinleri çiz
        y = 40
        for i, text in enumerate(texts):
            if i == 0:  # Başlık için farklı stil
                cv2.putText(img, text, (20, y), font, title_font_scale, (0, 255, 0), 3)
                y += line_height + 15
            else:
                cv2.putText(img, text, (20, y), font, font_scale, text_color, 2)
                y += line_height
        
        return img
        
    def draw_polygon(self, event, x, y, flags, param):
        """Poligon çizimi için fare olaylarını işle"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_pts.append((x, y))
            cv2.circle(self.img_display, (x, y), 3, (0, 255, 0), -1)
            if len(self.current_pts) > 1:
                cv2.line(self.img_display, self.current_pts[-2], self.current_pts[-1], (0, 255, 0), 2)
            self.drawing = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def start_drawing(self):
        """Çizim işlemini başlat"""
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image', self.window_width, self.window_height)
        cv2.setMouseCallback('Image', self.draw_polygon)
        
        self.mask_saved = False  # Mask kaydedildi bayrağını sıfırla
        
        while True:
            display_img = self.img_display.copy()
            display_img = self.draw_info_panel(display_img)  # Bilgi paneli ekle
            
            # Mevcut poligon noktalarını çiz
            if len(self.current_pts) > 0:
                for pt in enumerate(self.current_pts):
                    # Numarasız nokta çiz
                    cv2.circle(display_img, pt[1], 5, (0, 255, 0), -1)
            
            # Resmi göster
            cv2.imshow('Image', display_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):  # 'q' tuşuna basılırsa kaydet ve çık
                if len(self.polygons) > 0 or len(self.current_pts) > 2:
                    # Eğer en az 3 noktası olan tamamlanmamış bir poligon varsa, ekle
                    if len(self.current_pts) > 2:
                        cv2.line(self.img_display, self.current_pts[-1], self.current_pts[0], (0, 255, 0), 2)
                        self.polygons.append(self.current_pts)
                    
                    # Mask'ları kaydet ve çık
                    self.mask_saved = True
                    break
                else:
                    # Poligon yoksa uyarı göster
                    warning_img = display_img.copy()
                    cv2.putText(warning_img, "Hiç poligon oluşturulmadı! En az bir poligon çizin.", 
                                (50, self.window_height-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Image', warning_img)
                    cv2.waitKey(1500)  # 1.5 saniye uyarı göster
            
            elif key == 27:  # ESC tuşu ile iptal et ve çık
                self.polygons = []  # Tüm poligonları temizle
                break
                
            elif key == ord('f'):  # 'f' tuşu ile mevcut poligonu tamamla
                if len(self.current_pts) > 2:
                    cv2.line(self.img_display, self.current_pts[-1], self.current_pts[0], (0, 255, 0), 2)
                    self.polygons.append(self.current_pts)
                    self.current_pts = []
                else:
                    # Poligonda en az 3 nokta yoksa uyarı göster
                    warning_img = display_img.copy()
                    cv2.putText(warning_img, "Poligon oluşturmak için en az 3 nokta gerekli!", 
                                (50, self.window_height-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Image', warning_img)
                    cv2.waitKey(1500)  # 1.5 saniye uyarı göster
            
            elif key == ord('c'):  # 'c' tuşu ile mevcut noktaları temizle
                self.current_pts = []
                self.img_display = self.img_display_original.copy()
                # Önceki poligonları tekrar çiz
                for poly in self.polygons:
                    for i in range(len(poly)-1):
                        cv2.line(self.img_display, poly[i], poly[i+1], (0, 255, 0), 2)
                    cv2.line(self.img_display, poly[-1], poly[0], (0, 255, 0), 2)

        # Tüm pencereleri kapat
        cv2.destroyAllWindows()
        
        # Sadece poligonlar kaydedildiyse mask oluştur
        if self.mask_saved and len(self.polygons) > 0:
            return self.ask_save_location()
        else:
            print("Mask oluşturma iptal edildi.")
            return False
    
    def ask_save_location(self):
        """Kullanıcıya mask'ın kaydedileceği konumu sor"""

        directory = QFileDialog.getExistingDirectory(
            None, 
            "Mask'ı Nereye Kaydetmek İstiyorsunuz?",
            os.path.expanduser("~/Desktop"),  # Varsayılan konum olarak masaüstü
        )
        
        # Kullanıcı bir konum seçtiyse
        if directory:
            self.output_path = directory
            self.create_masks()
            return True
        else:
            print("Mask kaydetme iptal edildi.")
            return False

    def create_masks(self):
        """Çizilen poligonlardan mask dosyaları oluştur"""
        # Çıktı dizini yoksa oluştur
        os.makedirs(self.output_path, exist_ok=True)
        
        # Arka planda işlem için MaskCreatorWorker sınıfını kullan
        worker = MaskCreatorWorker(
            self.polygons, 
            self.width, 
            self.height, 
            self.new_width, 
            self.new_height, 
            self.output_path
        )
        worker.run()  # Doğrudan çalıştır, gerekirse QThread olarak da kullanılabilir
        
        # Mask dosyalarının oluşturulduğunu doğrula
        mask_path = os.path.join(self.output_path, "mask_main.png")
        if os.path.exists(mask_path):
            print(f"Poligon mask {mask_path} olarak kaydedildi")
            return True
        else:
            print("Hata: Mask dosyaları oluşturulamadı")
            return False


class MaskCreatorWorker:
    """Mask oluşturma işlemleri için ayrı bir sınıf"""
    
    def __init__(self, polygons, width, height, new_width, new_height, output_path):
        self.polygons = polygons
        self.width = width
        self.height = height
        self.new_width = new_width
        self.new_height = new_height
        self.output_path = output_path
        
    def run(self):
        """Mask dosyalarını oluştur ve kaydet"""
        try:
            # Orijinal resim boyutunda bir mask oluştur
            mask = np.zeros((self.height, self.width), dtype=np.uint8)

            # Noktaları orijinal resim boyutuna ölçekle ve poligonları doldur
            for polygon in self.polygons:
                scaled_polygon = [(int(x * self.width / self.new_width), int(y * self.height / self.new_height)) for x, y in polygon]
                pts = np.array(scaled_polygon, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
            
            # Resmin alt kısmını otomatik olarak maskele (son %20'lik kısmı)
            bottom_height = int(self.height * 0.2)  # Alt %20'lik kısım
            bottom_rect = np.array([
                [0, self.height - bottom_height],
                [0, self.height],
                [self.width, self.height],
                [self.width, self.height - bottom_height]
            ], np.int32)
            bottom_rect = bottom_rect.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [bottom_rect], 255)

            # Çıktı dizini yoksa oluştur
            os.makedirs(self.output_path, exist_ok=True)

            # Orijinal mask'ı çıktı olarak kaydet
            output = mask.copy()

            # Maskları kaydet
            cv2.imwrite(f'{self.output_path}/mask_main.png', output, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            
            print(f"Masklar {self.output_path} dizinine kaydedildi.")
            return True
        except Exception as e:
            print(f"Mask oluşturma hatası: {str(e)}")
            return False