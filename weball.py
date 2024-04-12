import random
import pygame
import cv2
import numpy as np
import time
from cvzone.PoseModule import PoseDetector
from ultralytics import YOLO
import cv2
import math
import numpy as np
import torch
import button
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos

class ShotDetector:
    def __init__(self, screen, model, class_names, detector):
        # Load the YOLO model
        self.model = model
        self.class_names = class_names

        pygame.display.set_caption("Train Shooting")
        self.window = screen
        self.width = self.window.get_rect().width
        self.height = self.window.get_rect().height
        self.fps = 30
        self.clock = pygame.time.Clock()

        # Use webcam - 0 is default camera
        self.cap = cv2.VideoCapture(0)
        # self.cap = cv2.VideoCapture("videos/shooting1.mp4")
        self.cap.set(3, 1920)
        self.cap.set(4, 1080)

        # Initialize the PoseDetector class with the given parameters
        self.detector = detector

        self.ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Used to detect shots (upper and lower region)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Button images
        resume_img = pygame.image.load("images/button_resume.png").convert_alpha()
        quit_img = pygame.image.load("images/button_quit.png").convert_alpha()
        menu_img = pygame.image.load("images/menu_img.png").convert_alpha()
        quit_rect = quit_img.get_rect(center = (self.window.get_rect().centerx, self.window.get_rect().centery+100))
        resume_rect = resume_img.get_rect(center = (self.window.get_rect().centerx, self.window.get_rect().centery-50))
        self.quit_button = button.Button(quit_rect.left, quit_rect.top, quit_img, 1)
        self.resume_button = button.Button(resume_rect.left, resume_rect.top, resume_img, 1)
        self.menu_button = button.Button(50, 50, menu_img, 1)

        # Variables
        self.in_hand = False
        self.elbow_angle = 180
        self.shoulder_angle = 180
        self.hip_angle = 180
        self.knee_angle = 180
        self.ankle_angle = 180
        self.release_angle = 0
        self.shooting_leg = False
        self.rim_left = False
        self.results = {}
        self.start_time = time.time()
        self.total_time = 60
        self.game_paused = False
        self.start = True

        self.run()

    def run(self):
        while self.start:
            # Get Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.start = False
                    pygame.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.start = False
                    if event.key == pygame.K_ESCAPE:
                        self.game_paused = not self.game_paused
            
            if not self.game_paused:
                # Remained time is total time subtracted time spent since start time
                time_remain = int(self.total_time - (time.time()-self.start_time))
            else:
                # Update start time during pause to "freeze" time and keep time remaining unchanged
                self.start_time = time.time()-int(time.time()-self.start_time)

            # Timer is finished show results analysis
            if time_remain < 0:
                data = {
                    "fg": round(100*self.makes/self.attempts) if self.attempts > 0 else 0,
                    "attempts": self.results
                    }
                self.cap.release()
                TrainingAnalysis(self.window, data, True)
                break
            else:
                ret, self.frame = self.cap.read()

                if not ret:
                    # End of the video or an error occurred
                    break

                if not self.game_paused:
                    # Detect object using trained YOLO model to each frame
                    results = self.model(source=self.frame, stream=True, device="gpu", verbose=False)

                    # Find the human pose in the frame
                    self.frame = self.detector.findPose(self.frame, draw=False)

                    # Find the landmarks, bounding box, and center of the body in the frame
                    lmList, bboxInfo = self.detector.findPosition(self.frame, draw=False, bboxWithHands=False)

                    # Check if any body landmarks are detected
                    if lmList:
                        elbow_angle, img = self.detector.findAngle(lmList[11][0:2] if not self.rim_left else lmList[15][0:2],
                                                        lmList[13][0:2],
                                                        lmList[15][0:2] if not self.rim_left else lmList[11][0:2],)
                        
                        knee_angle, img = self.detector.findAngle(lmList[27][0:2] if not self.rim_left else lmList[23][0:2],
                                                        lmList[25][0:2],
                                                        lmList[23][0:2] if not self.rim_left else lmList[27][0:2],)

                        for r in results:
                            boxes = r.boxes
                            for box in boxes:
                                # Bounding box
                                x1, y1, x2, y2 = box.xyxy[0]
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                w, h = x2 - x1, y2 - y1

                                # Confidence
                                conf = math.ceil((box.conf[0] * 100)) / 100

                                # Class Name
                                cls = int(box.cls[0])
                                current_class = self.class_names[cls]

                                center = (int(x1 + w / 2), int(y1 + h / 2))

                                # Only create ball points if high confidence or near hoop
                                if (conf > .3 or (in_hoop_region(center, self.hoop_pos) and conf > 0.15)) and current_class == "basketball" and not (x1-15 < lmList[0][0] < x2+15 and y1-15 < lmList[0][1] < y2+15):
                                    self.ball_pos.append((center, self.frame_count, w, h, conf))
                                    if (x1-30 < lmList[19][0] < x2+30 and y1-30 < lmList[19][1] < y2+30):
                                        self.in_hand = True
                                    else:
                                        if lmList[13][1] < lmList[11][1]:
                                            if self.in_hand:
                                                self.release_angle = math.degrees(math.atan2((lmList[11][1] if not self.rim_left else center[1]) - lmList[11][1], (lmList[11][0]+100 if not self.rim_left else center[0])  - lmList[11][0]) -
                                                                              math.atan2((center[1] if not self.rim_left else lmList[11][1]) - lmList[11][1], (center[0] if not self.rim_left else lmList[11][0]-100) - lmList[11][0]))
                                                if (self.release_angle < 0):
                                                    self.release_angle += 360
                                        self.in_hand = False

                                # If elbow is above the shoulder then track kinematic variables
                                if lmList[13][1] < lmList[11][1]:
                                    if (self.elbow_angle > elbow_angle):
                                        self.elbow_angle = elbow_angle
                                    if (self.knee_angle > knee_angle):
                                        self.knee_angle = knee_angle
                                        self.shooting_leg = lmList[31][0] >= lmList[32][0] if not self.rim_left else lmList[31][0] <= lmList[32][0]
                                        self.ankle_angle, img = self.detector.findAngle(lmList[25][0:2] if not self.rim_left else lmList[31][0:2],
                                                        lmList[29][0:2],
                                                        lmList[31][0:2] if not self.rim_left else lmList[25][0:2])
                                        self.hip_angle, img = self.detector.findAngle(lmList[11][0:2] if not self.rim_left else lmList[25][0:2],
                                                        lmList[23][0:2],
                                                        lmList[25][0:2] if not self.rim_left else lmList[11][0:2])
                                        self.shoulder_angle, img = self.detector.findAngle(lmList[13][0:2] if not self.rim_left else lmList[23][0:2],
                                                        lmList[11][0:2],
                                                        lmList[23][0:2] if not self.rim_left else lmList[13][0:2])
                                # Create hoop points if high confidence
                                if conf > .5 and current_class == "rim":
                                    self.hoop_pos.append((center, self.frame_count, w, h, conf))
                                    self.rim_left = x1 < self.cap.get(3)/2
                
                # Detect shot, clean storage and increment frame counter
                self.clean_motion()
                self.shot_detection()
                self.frame_count += 1

                # Update frame
                img_RGB = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                img_RGB = np.rot90(img_RGB)
                frame = pygame.surfarray.make_surface(img_RGB).convert()
                frame = pygame.transform.flip(frame, True, False)
                self.window.blit(frame, (0, 0))

                # Game paused then show in-game menu
                if self.game_paused == True:
                    pause_img = self.window.copy()
                    pygame.draw.rect(pause_img, (0, 0, 0, 0),  (0, 0, 1920,1080))
                    pause_img.set_alpha(127)
                    self.window.blit(pause_img, (0,0))
                    if self.resume_button.draw(self.window):
                        self.game_paused = False
                    if self.quit_button.draw(self.window):
                        self.start = False
                else:
                    if self.menu_button.draw(self.window):
                        self.game_paused = True

                # Print timer and score
                font = pygame.font.Font(None, 144)
                center = self.window.get_rect().center
                text_score = font.render(f'{self.makes}/{self.attempts}', True, (255, 255, 255))
                minutes = time_remain // 60
                seconds = time_remain % 60
                text_time = font.render("{:01d}:{:02d}".format(minutes, seconds), True, (255, 255, 255))
                text_score_rect = text_score.get_rect(center = (center[0], int(10*self.height/11)))
                text_time_rect = text_time.get_rect(center = (int(self.width/10), int(10*self.height/11)))
                self.window.blit(text_score, text_score_rect)
                self.window.blit(text_time, text_time_rect)

            # Update Display
            pygame.display.update()
            # Set FPS
            self.clock.tick(self.fps)

    def clean_motion(self):
        # Clean and display ball motion
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)

        # Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            # Detecting when ball is in 'up' and 'down' area - ball can only be in 'down' area after it is in 'up'
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            # If ball goes from 'up' area to 'down' area in that order, increase attempt and reset
            if self.frame_count % 10 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.attempts += 1
                    self.up = False
                    self.down = False
                    scored = False

                    # If it is a make
                    if score(self.ball_pos, self.hoop_pos):
                        scored = True
                        self.makes += 1
                    
                    self.results[self.attempts] = {
                        "scored": scored,
                        "elbow_angle": round(self.elbow_angle),
                        "shoulder_angle": round(self.shoulder_angle),
                        "hip_angle": round(self.hip_angle),
                        "knee_angle": round(self.knee_angle),
                        "ankle_angle": round(self.ankle_angle),
                        "release_angle": round(self.release_angle),
                        "shooting_leg": self.shooting_leg
                    }
                    
                    # Reset angle values
                    self.elbow_angle=180
                    self.knee_angle=180
                    self.release_angle=0
                    self.shooting_leg=False

class DribbleDetector:
    def __init__(self, screen, model, class_names, detector):
        # Load the YOLO model created from main.py - change text to your relative path
        self.model = model
        self.class_names = class_names

        pygame.display.set_caption("Train dribbling")
        self.window = screen
        self.width = self.window.get_rect().width
        self.height = self.window.get_rect().height
        self.fps = 30
        self.clock = pygame.time.Clock()

        # Use webcam - 0 is default camera
        self.cap = cv2.VideoCapture(0)
        # self.cap = cv2.VideoCapture("videos/dribbling.mp4")
        self.cap.set(3, 1920)
        self.cap.set(4, 1080)

        # Initialize the PoseDetector class with the given parameters
        self.detector = detector
        
        # Showing up ball image
        self.ball_img = pygame.image.load('images/basketball_dribble.png').convert_alpha()
        self.ball_rect = self.ball_img.get_rect()
        self.ball_rect.x, self.ball_rect.y = 700, 500

        # Load images and define positions
        resume_img = pygame.image.load("images/button_resume.png").convert_alpha()
        quit_img = pygame.image.load("images/button_quit.png").convert_alpha()
        menu_img = pygame.image.load("images/menu_img.png").convert_alpha()
        quit_rect = quit_img.get_rect(center = (self.window.get_rect().centerx, self.window.get_rect().centery+100))
        resume_rect = resume_img.get_rect(center = (self.window.get_rect().centerx, self.window.get_rect().centery-50))

        # Set buttons
        self.quit_button = button.Button(quit_rect.left, quit_rect.top, quit_img, 1)
        self.resume_button = button.Button(resume_rect.left, resume_rect.top, resume_img, 1)
        self.menu_button = button.Button(50, 50, menu_img, 1)

        # Variables
        self.misses = 0
        self.makes = 0
        self.start_time = time.time()
        self.total_time = 60
        self.frame = None
        self.in_left_hand = False
        self.in_right_hand = False
        self.left_shoulder = [0, 0]
        self.right_shoulder = [0, 0]
        self.hand_length = 300
        self.checkpoint_timer = time.time()
        self.game_paused = False
        self.start = True

        self.run()

    def run(self):
        while self.start:
            # Get Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.start = False
                    pygame.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.start = False
                    if event.key == pygame.K_ESCAPE:
                        self.game_paused = not self.game_paused
                
            # Apply Logic
            if not self.game_paused:
                time_remain = int(self.total_time - (time.time()-self.start_time))
            else:
                self.start_time = time.time()-int(time.time()-self.start_time)
                self.checkpoint_timer = time.time()-int(time.time()-self.checkpoint_timer)
            
            # Timer is finished show results analysis
            if time_remain < 0:
                data = {
                    "makes":self.makes,
                    "misses":self.misses
                }
                self.cap.release()
                TrainingAnalysis(self.window, data, False)
                break
            else:
                ret, self.frame = self.cap.read()

                if not ret:
                    # End of the video or an error occurred
                    break
                
                if not self.game_paused:
                    results = self.model(source=self.frame, stream=True, device="gpu", verbose=False)

                    # Find the human pose in the frame
                    self.frame = self.detector.findPose(self.frame, draw=False)

                    # Find the landmarks, bounding box, and center of the body in the frame
                    lmList, bboxInfo = self.detector.findPosition(self.frame, draw=False, bboxWithHands=False)

                    ball_life_span = int(time.time()-self.checkpoint_timer)

                    if ball_life_span >= 5:
                        self.checkpoint_timer = time.time()
                        self.resetBalloon()
                        self.misses += 1

                    # Check if any body landmarks are detected
                    if lmList:
                        length, img, info = self.detector.findDistance(lmList[11][0:2],
                                                                        lmList[15][0:2],
                                                                        img=None,
                                                                        color=(255, 0, 0),
                                                                        scale=0)
                        self.hand_length = max(self.hand_length, int(length))
                        for r in results:
                            boxes = r.boxes
                            for box in boxes:
                                # Bounding box
                                x1, y1, x2, y2 = box.xyxy[0]
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                w, h = x2 - x1, y2 - y1

                                # Confidence
                                conf = math.ceil((box.conf[0] * 100)) / 100

                                # Class Name
                                cls = int(box.cls[0])
                                current_class = self.class_names[cls]

                                center = (int(x1 + w / 2), int(y1 + h / 2))

                                # Only create ball points if high confidence or near hoop
                                if conf > .3 and current_class == "basketball" and not (x1-15 < lmList[0][0] < x2+15 and y1-15 < lmList[0][1] < y2+15):
                                    if (x1-10 < lmList[19][0] < x2+10 and y1-10 < lmList[19][1] < y2+10):
                                        self.in_left_hand = False
                                        self.in_right_hand = True
                                        self.right_shoulder = lmList[11][0:2]

                                    if (x1-10 < lmList[18][0] < x2+10 and y1-10 < lmList[18][1] < y2+10):
                                        self.left_shoulder = lmList[12][0:2]
                                        self.in_left_hand = True
                                        self.in_right_hand = False
                                    if self.ball_rect.colliderect(x1, y1, w, h):
                                        self.resetBalloon()
                                        self.makes += 1
                
                img_RGB = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                img_RGB = np.rot90(img_RGB)
                frame = pygame.surfarray.make_surface(img_RGB).convert()
                frame = pygame.transform.flip(frame, True, False)
                self.window.blit(frame, (0, 0))
                self.window.blit(self.ball_img, self.ball_rect)

                # Game paused then show in-game menu
                if self.game_paused == True:
                    pause_img = self.window.copy()
                    pygame.draw.rect(pause_img, (0, 0, 0, 0),  (0, 0, 1920,1080))
                    pause_img.set_alpha(127)
                    self.window.blit(pause_img, (0,0))
                    if self.resume_button.draw(self.window):
                        self.game_paused = False
                    if self.quit_button.draw(self.window):
                        self.start = False
                else:
                    if self.menu_button.draw(self.window):
                        self.game_paused = True

                # Update frame
                font = pygame.font.Font(None, 144)
                ballfont = pygame.font.Font(None, 64)
                text_score = font.render(f'{self.makes}/{self.makes+self.misses}', True, (255, 255, 255))
                minutes = time_remain // 60
                seconds = time_remain % 60
                text_time = font.render("{:01d}:{:02d}".format(minutes, seconds), True, (255, 255, 255))
                centerx = self.window.get_rect().centerx
                self.window.blit(text_score, (centerx, int(7*self.height/8)))
                self.window.blit(text_time, (int(self.height/10), int(7*self.height/8)))
                text = ballfont.render(str(5-ball_life_span), True, (255, 255, 255))
                text.set_alpha(150)
                text_rect = text.get_rect(center = self.ball_rect.center)
                self.window.blit(text, text_rect)
            
            # Update Display
            pygame.display.update()
            # Set FPS
            self.clock.tick(self.fps)

    def resetBalloon(self):
        self.checkpoint_timer = time.time()

        # Generate checkpoint on the opposite side of the hand
        if (self.in_right_hand):
            self.ball_rect.x = random.randint(max(300, int(self.left_shoulder[0]-self.hand_length)), max(300, int(self.left_shoulder[0]-self.hand_length/2)))
            self.ball_rect.y = random.randint(min(900, int(self.left_shoulder[1]+self.hand_length/4)), min(900, int(self.left_shoulder[1]+self.hand_length)))
        elif (self.in_left_hand):
            self.ball_rect.x = random.randint(min(1600, int(self.right_shoulder[0]+self.hand_length/2)), min(1600, int(self.right_shoulder[0]+self.hand_length)))
            self.ball_rect.y = random.randint(min(900, int(self.right_shoulder[1]+self.hand_length/4)), min(900,int(self.right_shoulder[1]+self.hand_length)))

class TrainingAnalysis:
    def __init__(self, screen, data, shooting):
        self.window = screen
        self.width = screen.get_rect().width
        self.results = data
        self.shooting = shooting

        # Load images and set buttons
        quit_img = pygame.image.load("images/button_quit.png").convert_alpha()
        feedback_img = pygame.image.load("images/button_options.png").convert_alpha()
        self.feedback_button = button.Button(1870-feedback_img.get_width(), 50, feedback_img, 1)
        self.quit_button = button.Button(50, 50, quit_img, 1)
        
        self.fps = 30
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Analysis")

        if shooting == True:
            self.shooting_analysis()
        else:
            self.dribbling_analysis()

    def shooting_analysis(self):
        # Optimal kinematic variables according to Cabarkapa et al. (2022) in their "Impact of Distance and Proficiency on Shooting Kinematics in Professional Male Basketball Players" research
        excellent_shot_ranges = {
            "knee": [123.1, 9.3],
            "hip": [145.5, 7.3],
            "ankle": [62.8, 6.5],
            "elbow": [57.8, 12.9],
            "shoulder": [74.6, 26.1],
            "release": [60.1, 5.6],
        }

        start = True
        feedback = False
        while start:
            self.window.fill((0,0,0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    start = False
                    pygame.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        start = False
            
            # If no attempts then print "try again"
            if len(self.results['attempts']) == 0:
                if self.quit_button.draw(self.window):
                    start = False
                font_fail = pygame.font.Font(None, 60)
                text_fail = font_fail.render("You have not tried any shot. Please, try again!", True, (255, 255, 255))
                text_fail_rect = text_fail.get_rect(center = (self.window.get_rect().centerx, self.window.get_rect().centery))
                self.window.blit(text_fail, text_fail_rect)
            else:
                # Create a dictionary with correct and wrong results
                data = {}
                for key, val in self.results["attempts"].items():
                        data[key] = {}
                        if val['scored'] == True:
                            data[key] = {"scored": ['scored', True]}
                        else:
                            data[key] = {"scored": ['missed', False]}
                        for name, range in excellent_shot_ranges.items():
                            if range[0]-range[1] <= val[f'{name}_angle'] <= range[0]+range[1]:
                                data[key][f'{name}'] = [val[f'{name}_angle'], True]
                            elif val[f'{name}_angle'] > range[0]+range[1]:
                                data[key][f'{name}'] = [val[f'{name}_angle'], False]
                            elif range[0]-range[1] > val[f'{name}_angle']:
                                data[key][f'{name}'] = [val[f'{name}_angle'], False]
                        data[key]['shooting leg'] = ["front", True] if val['shooting_leg'] == True else ["back", False]

                # If feedback is not chosen show results table otherwise analyze the results
                if not feedback:
                    if self.quit_button.draw(self.window):
                        start = False
                    if self.feedback_button.draw(self.window):
                        feedback = True
                    captions = ['', 'scored', 'knee°', 'hip°', 'ankle°', 'elbow°', 'shoulder°', 'release°', 'shooting leg']

                    length_captions = len(captions)
                    font = pygame.font.Font(None, 40)
                    pygame.draw.rect(self.window, (255, 255, 255), [100, 200, 1720, (len(data)+3)*60], 0, border_radius=10)

                    wide = 1
                    wide_unit = (self.width/(length_captions+2))
                    for caption in captions:
                        text_caption = font.render(caption, True, (0, 0, 0))
                        self.window.blit(text_caption, (int(wide*wide_unit), 250))
                        wide+=1

                    index = 1
                    length_data = len(data)
                    max_height = 650
                    height_unit = min(50, int(max_height/length_data))
                    for key, value in data.items():
                        wide = 1
                        text_attempt = font.render(f'Attempt #{key}', True, (0, 0, 0))
                        self.window.blit(text_attempt, (int(wide*wide_unit), 250 + height_unit*index))
                        wide+=1
                        for name, detail in value.items():
                            text_detail = font.render(f'{detail[0]}', True, (0, 255, 0) if detail[1] else (255, 0, 0))
                            self.window.blit(text_detail, (int(wide*wide_unit), 250 + height_unit*index))
                            wide+=1
                        index+=1
                    
                    wide = 1
                    index+=1
                    text_optimal = font.render(f'Optimal value', True, (0, 0, 0))
                    self.window.blit(text_optimal, (int(wide*wide_unit), 250 + height_unit*index))
                    wide+=2
                    for name, range in excellent_shot_ranges.items():
                        text_detail = font.render(f'{range[0]}±{range[1]}', True, (0, 0, 0))
                        self.window.blit(text_detail, (int(wide*wide_unit), 250 + height_unit*index))
                        wide+=1
                    text_optimal = font.render(f'front', True, (0, 0, 0))
                    self.window.blit(text_optimal, (int(wide*wide_unit), 250 + height_unit*index))

                    fg = self.results["fg"]
                    introduction = f"You've scored {fg}% of your shots."
                    
                    font_intro = pygame.font.Font(None, 60)
                    text_intro = font_intro.render(introduction, True, (255, 255, 255))
                    text_intro_rect = text_intro.get_rect(center = (self.window.get_rect().centerx, 150))
                    self.window.blit(text_intro, text_intro_rect)
                
                else:
                    pygame.draw.rect(self.window, (255, 255, 255), [250, 200, 1420, 700], 0, border_radius=10)
                    if self.quit_button.draw(self.window):
                        feedback = False
                    mistakes = {}
                    right = {}
                    sum = {}
                    for attempt, values in data.items():
                        for key, [value, is_correct] in values.items():
                            if key != 'scored':
                                sum[key] = sum.get(key, 0) + (value if not isinstance(value, str) else 1 if value == 'front' else -1)
                                if not is_correct:
                                    mistakes[key] = mistakes.get(key, 0) + (1 if values['scored'][1] else 2)
                                elif is_correct:
                                    right[key] = right.get(key, 0) + (2 if values['scored'][1] else 1)


                    top_mistakes = sorted(mistakes.items(), key=lambda x: x[1], reverse=True)[:3]
                    top_right = sorted(right.items(), key=lambda x: x[1], reverse=True)[:3]
                    font = pygame.font.Font(None, 50)

                    # Print best results
                    self.window.blit(font.render("Top results:", True, (0, 0, 0)), (300, 250))
                    index = 1
                    for key, val in top_right:
                        average_value = 'in front' if int(sum[key]/len(data)) == 1 else 'in back' if key=='shooting leg' else int(sum[key]/len(data))
                        text = font.render(f"- {key} average value: {average_value}", True, (0, 0, 0))
                        self.window.blit(text, (400, 250+50*index))
                        index+=1
                    self.window.blit(font.render("Good job! Keep it up!", True, (0, 0, 0)), (300, 250+50*index))

                    # Print worst results
                    self.window.blit(font.render("Frequent mistakes:", True, (0, 0, 0)), (300, 550))
                    index = 1
                    advice = ["You need to"]
                    for key, val in top_mistakes:
                        average_value = 'in front' if int(sum[key]/len(data)) == 1 else 'in back' if key=='shooting leg' else int(sum[key]/len(data))
                        low_excellent = int(excellent_shot_ranges[key][0]-excellent_shot_ranges[key][1])
                        high_excellent = int(excellent_shot_ranges[key][0]+excellent_shot_ranges[key][1])
                        if key != 'shooting leg' and average_value < low_excellent:
                            advice.append(f"increase {key} angle up to {low_excellent}")
                        elif key != 'shooting leg' and average_value > high_excellent:
                            advice.append(f"decrease {key} angle down to {high_excellent}")
                        elif key != 'shooting leg':
                            advice.append(f'maintain {key} angle between {low_excellent} and {high_excellent}')
                        else:
                            advice.append("put your shooting leg slightly in the front")
                        if index == 1:
                            advice[-1] += ','
                        elif index == 2:
                            advice[-1] += ' and'
                        else:
                            advice[-1] += '.'
                        text = font.render(f"- {key} average value: {average_value}", True, (0, 0, 0))
                        self.window.blit(text, (400, 550+50*index))
                        index+=1
                    
                    text = " ".join(advice)
                    wrapped_text = self.wrap_text(text, font, 1320)

                    for line in wrapped_text:
                        self.window.blit(font.render(line, True, (0, 0, 0)), (300, 550+50*index))
                        index+=1
                                
            # Update Display
            pygame.display.update()
            # Set FPS
            self.clock.tick(self.fps)

    def dribbling_analysis(self):
        run = True
        while run:
            makes = self.results['makes']
            misses = self.results['misses']
            self.window.fill((0,0,0))
            center = self.window.get_rect().center
            font = pygame.font.Font(None, 50)
            advice = ["You have"]
            if makes == 0:
                advice.append("not covered any ball")
                if misses > 0:
                    advice.append('and')
            elif makes == 1:
                advice.append("covered one ball")
                if misses > 0:
                    advice.append('but')
            elif makes > 1:
                advice.append(f"covered {makes} balls")
                if misses > 0:
                    advice.append('but')

            if misses == 0:
                advice.append("and did not miss any ball")
                if makes > 0:
                    comments = "Well done! You have showed your 100%"
            elif misses == 1:
                advice.append("missed one ball")
                if makes > 0:
                    comments = "Well done! You are almost there to get your 100%"
            elif misses > 1:
                advice.append(f"missed {misses} balls")
                if makes > 0:
                    comments = "Well done! Keep practicing and you will get your 100%"
            font_big = pygame.font.Font(None, 70)
            text_score = font_big.render(f'Your dribbling accuracy: {int(100*(makes/(makes+misses)))}%', True, (255, 255, 255))
            text_time = font.render(" ".join(advice), True, (255, 255, 255))
            textComment = font.render(comments, True, (255, 255, 255))
            textScore_rect = text_score.get_rect(center = (center[0], 400))
            textTime_rect = text_time.get_rect(center = (center[0], 500))
            textComment_rect = textComment.get_rect(center = (center[0], 550)) 
            self.window.blit(text_score, textScore_rect)
            self.window.blit(text_time, textTime_rect)
            self.window.blit(textComment, textComment_rect)
            if self.quit_button.draw(self.window):
                run = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        run = False
            pygame.display.update()
        
    def wrap_text(self, text, font, width):
        text_lines = text.replace('\t', '    ').split('\n')
        if width is None or width == 0:
            return text_lines

        wrapped_lines = []
        for line in text_lines:
            line = line.rstrip() + ' '
            if line == ' ':
                wrapped_lines.append(line)
                continue

            # Get the leftmost space ignoring leading whitespace
            start = len(line) - len(line.lstrip())
            start = line.index(' ', start)
            while start + 1 < len(line):
                # Get the next potential splitting point
                next = line.index(' ', start + 1)
                if font.size(line[:next])[0] <= width:
                    start = next
                else:
                    wrapped_lines.append(line[:start])
                    line = line[start+1:]
                    start = line.index(' ')
            line = line[:-1]
            if line:
                wrapped_lines.append(line)
        return wrapped_lines

class Menu:
    def __init__(self):
        # Initialize
        pygame.init()

        #create game window
        screen = pygame.display.set_mode((1920, 1080))
        center = screen.get_rect().center

        #set icon
        icon = pygame.image.load('images/weball.png')
        pygame.display.set_icon(icon) 

        #game variables
        game_paused = False

        torch.cuda.set_device(0) # Set to your desired GPU number

        # Load the YOLO model
        model = YOLO("train/weights/best.pt")
        class_names = ['basketball', 'people', 'rim']

        # Load the CVZone PoseDetector which use MediaPipe pose estimation
        detector = PoseDetector(staticMode=False,
                                modelComplexity=1,
                                smoothLandmarks=True,
                                enableSegmentation=False,
                                smoothSegmentation=True,
                                detectionCon=0.5,
                                trackCon=0.5)

        # Load button images
        quit_img = pygame.image.load("images/button_quit.png").convert_alpha()
        back_img = pygame.image.load('images/button_back.png').convert_alpha()
        weball_logo = pygame.image.load('images/weball.png').convert_alpha()
        dribbling_img = pygame.image.load('images/dribbling.png').convert_alpha()
        shooting_img = pygame.image.load('images/shooting.png').convert_alpha()
        menu_img = pygame.image.load("images/menu_img.png").convert_alpha()

        # Get rectangles for images to define center points
        weball_rect = weball_logo.get_rect(center=( int(2*screen.get_rect().width/7), center[1]))
        dribbling_rect = dribbling_img.get_rect(center = (int(5*screen.get_rect().width/7), int(2*screen.get_rect().height/7)))
        shooting_rect = shooting_img.get_rect(center = (int(5*screen.get_rect().width/7), int(5*screen.get_rect().height/7)))
        quit_rect = quit_img.get_rect(center = (center[0], center[1]+100))
        back_rect = back_img.get_rect(center = (center[0], center[1]-100))

        # Create button instances
        quit_button = button.Button(quit_rect.left, quit_rect.top, quit_img, 1)
        back_button = button.Button(back_rect.left, back_rect.top, back_img, 1)
        dribbling_button = button.Button(dribbling_rect.left, dribbling_rect.top, dribbling_img, 1)
        shooting_button = button.Button(shooting_rect.left, shooting_rect.top, shooting_img, 1)
        menu_button = button.Button(50, 50, menu_img, 1)

        # Game loop
        run = True
        while run:
            pygame.display.set_caption("Main Menu")
            screen.fill((0, 0, 0))
            screen.blit(weball_logo, weball_rect)

            if dribbling_button.draw(screen) and not game_paused:
                DribbleDetector(screen, model, class_names, detector)
            if shooting_button.draw(screen) and not game_paused:
                ShotDetector(screen, model, class_names, detector)
                
            if not game_paused:
                if menu_button.draw(screen):
                    game_paused = True
            else:
                work_img = screen.copy()
                pygame.draw.rect(work_img, (0, 0, 0, 0),  (0, 0, 1920,1080))
                work_img.set_alpha(127)
                screen.blit(work_img, (0,0))
                if quit_button.draw(screen):
                    run = False
                if back_button.draw(screen):
                    game_paused = False

        # Event handler
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        game_paused = not game_paused
                if event.type == pygame.QUIT:
                    run = False

            pygame.display.update()

        pygame.quit()

if __name__ == "__main__":
    Menu()