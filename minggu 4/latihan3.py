# ============================================
# REAL-TIME VIDEO ENHANCEMENT
# ============================================

import cv2
import numpy as np
import time


class RealTimeEnhancement:

    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.history_buffer = []
        self.buffer_size = 5

    # =====================================
    # FRAME ENHANCEMENT FUNCTION
    # =====================================

    def enhance_frame(self, frame, enhancement_type='adaptive'):

        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if enhancement_type == 'adaptive':

            # CLAHE (good for real-time)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

        elif enhancement_type == 'gamma':

            gamma = 0.7
            enhanced = np.power(gray/255.0, gamma)
            enhanced = np.uint8(enhanced*255)

        elif enhancement_type == 'sharpen':

            kernel = np.array([[0,-1,0],
                               [-1,5,-1],
                               [0,-1,0]])

            enhanced = cv2.filter2D(gray,-1,kernel)

        else:
            enhanced = gray

        # =====================================
        # TEMPORAL SMOOTHING (avoid flickering)
        # =====================================

        self.history_buffer.append(enhanced)

        if len(self.history_buffer) > self.buffer_size:
            self.history_buffer.pop(0)

        temporal_avg = np.mean(self.history_buffer, axis=0).astype(np.uint8)

        return temporal_avg


# ============================================
# MAIN PROGRAM
# ============================================

cap = cv2.VideoCapture(0)  # webcam

enhancer = RealTimeEnhancement(target_fps=30)

print("Press Q to quit")

while True:

    start_time = time.time()

    ret, frame = cap.read()

    if not ret:
        break

    # resize for faster processing
    frame = cv2.resize(frame, (640,480))

    enhanced_frame = enhancer.enhance_frame(frame, 'adaptive')

    # convert grayscale to BGR for display
    enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)

    # show results
    cv2.imshow("Original Video", frame)
    cv2.imshow("Enhanced Video", enhanced_frame)

    # =====================================
    # FPS CONTROL
    # =====================================

    elapsed = time.time() - start_time
    delay = max(1, int((1/enhancer.target_fps - elapsed)*1000))

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()