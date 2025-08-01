import cv2
from ultralytics import YOLO
from pynput import keyboard
from djitellopy import Tello
import time

class model:

  def __init__(self, model):
    
    self.model = YOLO(model)

  def realTimePredict(self, device, confidence):

    video_capture = cv2.VideoCapture(device)

    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Real-time video capture loop
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Run YOLOv11 inference
        results = self.model(frame, imgsz = 640)[0]

        # Draw detection results
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result[:6]
            if conf > confidence:
                label = f'{self.model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow('YOLOv11 Real-Time Detection', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()
  
  def videoPredict(self, videoPath, confidence):
      input_video_path = videoPath
      output_video_path = 'out.mp4'

      # Open the video using OpenCV
      video_capture = cv2.VideoCapture(input_video_path)

      # Get video properties
      frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
      frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
      fps = int(video_capture.get(cv2.CAP_PROP_FPS))
      total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

      # Define the codec and create VideoWriter object to save output video
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
      out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

      # Iterate over each frame
      frame_count = 0
      while video_capture.isOpened():
          ret, frame = video_capture.read()  # Read a frame
          if not ret:
              break
          
          # Apply YOLOv11 object detection
          results = self.model(frame)[0]
          
          # Iterate through the detections and draw bounding boxes
          for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result[:6]
            if conf > confidence:
                label = f'{self.model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
          
          # Write the processed frame to the output video
          out_video.write(frame)
          
          # Print progress
          frame_count += 1
          print(f'Processed frame {frame_count}/{total_frames}')

          # Release resources
      video_capture.release()
      out_video.release()
      cv2.destroyAllWindows()

      print(f'Output video saved to {output_video_path}')

  def getBattery(self):
      tello = Tello()
      tello.connect()
      print(f"Battery: {tello.get_battery()}%")
  
  def realTimeDrone(self, confidence):
     
     # Initialize Tello
      tello = Tello()
      tello.connect()
      print(f"Battery: {tello.get_battery()}%")
      tello.streamon()
      frame_reader = tello.get_frame_read()

      # State
      flying = False
      running = True

      # Control function
      def on_press(key):
          nonlocal flying, running
          try:
              k = key.char.lower()
          except:
              k = str(key)

          if k == 'w':
              tello.move_forward(30)
          elif k == 's':
              tello.move_back(30)
          elif k == 'a':
              tello.move_left(20)
          elif k == 'd':
              tello.move_right(30)
          elif k == 'q':
              tello.rotate_counter_clockwise(45)
          elif k == 'e':
              tello.rotate_clockwise(45)
          elif k == 'r':
              tello.move_up(20)
          elif k == 'f':
              tello.move_down(20)
          elif k == 't' and not flying:
              tello.takeoff()
              flying = True
          elif k == 'l' and flying:
              tello.land()
              flying = False
          elif k == 'z':
              tello.flip_forward()
          elif k == 'x':
              tello.flip_back()
          elif k == 'c':
              tello.flip_left()
          elif k == 'v':
              tello.flip_right()
          elif k == 'b':
              print(f"Battery: {tello.get_battery()}%")
          elif k == 'Key.esc':
              print("Exiting...")
              running = False
              if flying:
                  tello.land()
              tello.end()
              return False  # Stop listener

      # Start keyboard listener in a separate thread
      listener = keyboard.Listener(on_press=on_press)
      listener.start()

      print("""
      Controls:
      T = takeoff
      L = land
      W/A/S/D = move
      R/F = up/down
      Q/E = rotate
      Z/X/C/V = flips
      B = battery
      ESC = quit
      """)

      # Detection + display loop
      try:
          while running:  
              # Get and convert frame
              frame = frame_reader.frame
              frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
              frame = cv2.resize(frame, (640, 480))

              # YOLO prediction
              results = self.model(frame, imgsz=640)[0]

              for result in results.boxes.data.tolist():
                  x1, y1, x2, y2, conf, cls = result[:6]
                  if conf > confidence:
                      label = f'{self.model.names[int(cls)]} {conf:.2f}'
                      cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                      cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

              cv2.imshow('YOLOv11 Real-Time Detection', frame)
              if cv2.waitKey(1) & 0xFF == ord('q'):
                  running = False
      finally:
          if flying:
              tello.land()
          tello.streamoff()
          tello.end()
          cv2.destroyAllWindows()
          listener.stop()

  def pursueObject(self, confidence):

    tello = Tello()
    tello.connect()
    print(f"Battery: {tello.get_battery()}%")

    tello.streamon()
    time.sleep(1)  # Let camera warm up
    frame_reader = tello.get_frame_read()

    # State
    flying = False
    running = True

    # Control function
    def on_press(key):
        nonlocal flying, running
        try:
            k = key.char.lower()
        except:
            k = str(key)

        if k == 't' and not flying:
            tello.takeoff()
            tello.move_down(40)
            flying = True
        elif k == 'l' and flying:
            tello.land()
            flying = False
        elif k == 'Key.esc':
            print("Exiting...")
            running = False
            if flying:
                tello.land()
            tello.end()
            return False  # Stop listener

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("""
    Controls:
    T = takeoff
    L = land
    ESC = quit
    """)

    # Detection loop
    try:
        while running:
            frame = frame_reader.frame
            if frame is None:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (640, 480))

            results = self.model(frame, imgsz=640)[0]

            frame_center_x = 640 // 2
            frame_center_y = 480 // 2
            move_threshold = 40  # tolerance
            speed = 20

            # Default no movement
            left_right = 0
            up_down = 0
            forward_back = 0
            yaw = 0

            # Track the most confident box only
            if results.boxes.data.tolist():
                top_result = sorted(results.boxes.data.tolist(), key=lambda x: x[4], reverse=True)[0]
                x1, y1, x2, y2, conf, cls = top_result[:6]

                if conf > confidence:
                    object_center_x = int((x1 + x2) / 2)
                    object_center_y = int((y1 + y2) / 2)

                    # Horizontal adjustment
                    if object_center_x < frame_center_x - move_threshold:
                        left_right = -speed
                    elif object_center_x > frame_center_x + move_threshold:
                        left_right = speed

                    # Vertical adjustment
                    if y2 < 380:
                        forward_back = speed  # move forward   
                    elif y2 > 450:  # adjust threshold based on test
                        forward_back = -speed  # move back



                    # Draw rectangle
                    label = f'{self.model.names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Send smoothed RC control
            if flying:
                tello.send_rc_control(left_right, forward_back, up_down, yaw)

            cv2.imshow('YOLOv11 Real-Time Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False

            time.sleep(0.1)  # reduce command spam
    finally:
        if flying:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()
        listener.stop()
