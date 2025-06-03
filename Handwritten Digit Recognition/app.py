import pygame
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2
import sys
from scipy.ndimage import center_of_mass

# Constants
wsx, wsy = 640, 480
boundry = 5
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
gray = (100, 100, 100)

# Load CNN model
model = load_model("bestmodel.h5")
labels = {i: str(i) for i in range(10)}

# Initialize Pygame
pygame.init()
FONT = pygame.font.SysFont("Arial", 18)
DISPLAYSURF = pygame.display.set_mode((wsx, wsy))
pygame.display.set_caption("Digit Recognizer")

# State variables
iswriting = False
numberxcord = []
numberycord = []
prediction_count = 0

def draw_clear_button():
    pygame.draw.rect(DISPLAYSURF, gray, (10, 10, 100, 40))
    text = FONT.render("Clear (N)", True, white)
    DISPLAYSURF.blit(text, (20, 18))

def center_image(image):
    cy, cx = center_of_mass(image)
    if np.isnan(cx) or np.isnan(cy):
        return image  # avoid error on empty input
    shiftx = int(np.round(14 - cx))
    shifty = int(np.round(14 - cy))
    return np.roll(np.roll(image, shiftx, axis=1), shifty, axis=0)

def predict_digit(img_arr):
    image = cv2.resize(img_arr, (28, 28))
    image = np.pad(image, ((10, 10), (10, 10)), 'constant', constant_values=0)
    image = cv2.resize(image, (28, 28))
    image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)[1]
    image = center_image(image)
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    prediction = model.predict(image)
    return prediction

# Main loop
while True:
    draw_clear_button()

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            x, y = event.pos
            if x > 0 and y > 0:
                pygame.draw.circle(DISPLAYSURF, white, (x, y), 8, 0)
                numberxcord.append(x)
                numberycord.append(y)

        if event.type == MOUSEBUTTONDOWN:
            x, y = event.pos
            if 10 <= x <= 110 and 10 <= y <= 50:
                DISPLAYSURF.fill(black)
            else:
                iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if numberxcord and numberycord:
                minx = max(min(numberxcord) - boundry, 0)
                maxx = min(max(numberxcord) + boundry, wsx)
                miny = max(min(numberycord) - boundry, 0)
                maxy = min(max(numberycord) + boundry, wsy)

                # Extract drawing
                img_arr = pygame.surfarray.array3d(DISPLAYSURF)[minx:maxx, miny:maxy]
                img_arr = cv2.cvtColor(np.transpose(img_arr, (1, 0, 2)), cv2.COLOR_RGB2GRAY)

                if img_arr.size > 0:
                    prediction = predict_digit(img_arr)
                    confidence = np.max(prediction)
                    label = np.argmax(prediction)
                    text = f"{labels[label]} ({confidence*100:.1f}%)" if confidence > 0.7 else "Uncertain"

                    textsurface = FONT.render(text, True, red, white)
                    rect = textsurface.get_rect()
                    rect.left, rect.top = minx, maxy
                    DISPLAYSURF.blit(textsurface, rect)
                    prediction_count += 1

            # Reset coordinates
            numberxcord = []
            numberycord = []

        if event.type == KEYDOWN:
            if event.unicode.lower() == 'n':
                DISPLAYSURF.fill(black)

    pygame.display.update()
