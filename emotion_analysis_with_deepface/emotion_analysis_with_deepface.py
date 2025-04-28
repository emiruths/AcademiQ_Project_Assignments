import cv2
from deepface import DeepFace

# Kamerayı başlat
cap = cv2.VideoCapture(1)

# Sonsuz döngü: her kareyi işle
while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Yüzleri ve duyguları tespit et
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Eğer birden fazla yüz varsa, liste döner
        if isinstance(result, list):
            for face in result:
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                emotion = face['dominant_emotion']

                # Yüz etrafına dikdörtgen çiz
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Duyguyu yaz
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        else:
            # Tek bir yüz varsa
            x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
            emotion = result['dominant_emotion']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    except Exception as e:
        print("Hata:", e)

    # Kameradan alınan görüntüyü göster
    cv2.imshow('Duygu Analizi', frame)

    # q tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()
