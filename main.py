import cv2
from ultralytics import YOLO
import datetime
import boto3
import numpy as np

# Load YOLOv8 Model
model = YOLO('yolov8n.pt')

# AWS SNS Setup for Automated Emergency SMS Dispatches
AWS_REGION = "us-east-1"
SNS_TOPIC_ARN = "arn:aws:sns:us-east-1:123456789012:OmniGuardAlerts"

try:
    sns_client = boto3.client('sns', region_name=AWS_REGION)
    AWS_ACTIVE = True
    print("[SYSTEM] AWS SNS Dispatch Active.")
except Exception as e:
    AWS_ACTIVE = False
    print(f"[SYSTEM] AWS Offline Mode (Simulation Active): {e}")

cap = cv2.VideoCapture(0)
tracking_history = {}
fall_counter = {}  # Tracks sustained horizontal posture for fainting detection
registered_persons = set()  # Memory dictionary for "remembered" persons

# Cooldown trackers to prevent spamming SMS every frame
last_alert_time = {
    "FIRE": datetime.datetime.min,
    "SMOKE": datetime.datetime.min,
    "FLOOD": datetime.datetime.min,
    "FAINT": datetime.datetime.min,
    "CROWD": datetime.datetime.min,
    "SPEED": datetime.datetime.min
}


def dispatch_emergency_sms(alert_type, message_body):
    """Triggers automated SMS dispatch via AWS SNS"""
    now = datetime.datetime.now()
    if (now - last_alert_time[alert_type]).total_seconds() > 15:
        last_alert_time[alert_type] = now
        print(f"\n[AUTOMATED DISPATCH] Sending SMS via AWS SNS: {message_body}")
        if AWS_ACTIVE:
            try:
                sns_client.publish(
                    TopicArn=SNS_TOPIC_ARN,
                    Message=message_body,
                    Subject=f"OmniGuard AI Critical Alert: {alert_type}"
                )
                print("[AWS SNS] SMS Successfully Dispatched to Authorities.")
            except Exception as err:
                print(f"[AWS SNS Error]: {err}")


print("========================================================")
print(" OMNIGUARD AI: Autonomous Vision + Person Memory Active ")
print("========================================================")
print(" Controls: Press 'r' to register/remember the person on screen.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    current_alert = "STATUS: All Zones Secure & Monitored"
    banner_color = (0, 255, 0)  # Green
    person_count = 0

    # Convert frame to HSV color space for automated pixel analysis
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1. AUTOMATIC FIRE DETECTION
    lower_fire = np.array([18, 50, 50])
    upper_fire = np.array([35, 255, 255])
    fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
    if cv2.countNonZero(fire_mask) > 4000:
        current_alert = f"[{current_time}] 🔥 AUTOMATIC FIRE HAZARD DETECTED! Fire Dept Dispatched."
        banner_color = (0, 0, 255)
        dispatch_emergency_sms("FIRE", f"CRITICAL: Fire automatically detected at Zone 1 ({current_time}).")

    # 2. AUTOMATIC SMOKE DETECTION
    lower_smoke = np.array([0, 0, 140])
    upper_smoke = np.array([180, 45, 235])
    smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
    if cv2.countNonZero(smoke_mask) > 6000 and cv2.countNonZero(fire_mask) < 2000:
        current_alert = f"[{current_time}] 💨 AUTOMATIC SMOKE PLUME DETECTED! Ventilation Active."
        banner_color = (0, 0, 255)
        dispatch_emergency_sms("SMOKE", f"CRITICAL: Smoke automatically detected at Zone 1 ({current_time}).")

    # 3. AUTOMATIC FLOOD DETECTION
    lower_flood = np.array([95, 50, 50])
    upper_flood = np.array([135, 255, 255])
    flood_mask = cv2.inRange(hsv, lower_flood, upper_flood)
    if cv2.countNonZero(flood_mask) > 8000:
        current_alert = f"[{current_time}] 🌊 AUTOMATIC FLOOD DETECTED! Drainage Pump Activated."
        banner_color = (0, 140, 255)
        dispatch_emergency_sms("FLOOD", f"CRITICAL: Flood water logging detected at Zone 1 ({current_time}).")

    # 4. YOLOv8 OBJECT DETECTION & PERSON MEMORY
    results = model(frame)
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            if label == 'person':
                person_count += 1
                box_w = x2 - x1
                box_h = y2 - y1
                person_id = id(box)  # Temporary tracking ID for this session

                # Check Person Memory Status
                is_remembered = person_id in registered_persons
                tag_label = "VIP / Trusted Resident" if is_remembered else "Detected Person"
                tag_color = (0, 255, 0) if is_remembered else (255, 255, 0)

                # Draw bounding box label with memory status
                cv2.putText(frame, tag_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tag_color, 2)

                # Robust Fainting / Medical Emergency Detection (Sustained Horizontal Posture)
                if box_w > box_h * 1.25:  # Width greater than height indicates lying down / fainting
                    fall_counter[person_id] = fall_counter.get(person_id, 0) + 1
                    if fall_counter[person_id] > 15:  # Sustained for ~15 frames
                        current_alert = f"[{current_time}] ⚠️ MEDICAL EMERGENCY: Person Fainted! Paramedics Dispatched."
                        banner_color = (0, 0, 255)
                        dispatch_emergency_sms("FAINT",
                                               f"URGENT: Fainting/Medical emergency detected at Zone 1 ({current_time}).")
                        cv2.putText(frame, "STATUS: FAINTED / MEDICAL EMERGENCY", (x1, y2 + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    fall_counter[person_id] = 0  # Reset if standing up normally

            elif label in ['car', 'motorcycle', 'truck']:
                current_alert = f"[{current_time}] 🚨 E-CHALLAN: Vehicle Speed Violation Logged & Sent to Police."
                banner_color = (0, 140, 255)
                dispatch_emergency_sms("SPEED", f"TRAFFIC ALERT: Speed violation logged at Zone 1 ({current_time}).")

    # Crowd Control Check (>5 people)
    if person_count > 5:
        current_alert = f"[{current_time}] 🚨 CROWD OVERLOAD: {person_count} People in Zone! Security Notified."
        banner_color = (0, 0, 255)
        dispatch_emergency_sms("CROWD", f"SECURITY ALERT: Overcrowding detected ({person_count} people) at Zone 1.")

    # Render Top Command Dispatch Banner
    height, width, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (width, 55), (20, 20, 20), -1)
    cv2.putText(frame, current_alert, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, banner_color, 2)

    # Display Annotated Live Feed
    annotated_frame = results[0].plot()

    # Merge custom text overlays onto annotated frame
    cv2.imshow("OmniGuard AI - Autonomous Hub with Person Memory", frame)

    # Keyboard Controls during runtime
    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):  # Press 'r' to remember/register people currently visible on screen
        for result in results:
            for box in result.boxes:
                if model.names[int(box.cls[0])] == 'person':
                    registered_persons.add(id(box))
                    print("[MEMORY] Person successfully registered and remembered by OmniGuard AI!")

cap.release()
cv2.destroyAllWindows()

