from pygame.locals import KEYDOWN, K_ESCAPE, K_q
import pygame
import cv2
import sys

camera = cv2.VideoCapture(1)
success, img =  camera.read()
print(img.shape[1])
pygame.init()
pygame.display.set_caption("OpenCV camera stream on Pygame")
screen = pygame.display.set_mode([img.shape[1],img.shape[0]])
clock = pygame.time.Clock()


try:
    while True:
        clock.tick(60)
        ret, frame = camera.read()

        screen.fill([0, 0, 0])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.swapaxes(0, 1)
        pygame.surfarray.blit_array(screen, frame)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE or event.key == K_q:
                    sys.exit(0)

except (KeyboardInterrupt, SystemExit):
    pygame.quit()
    cv2.destroyAllWindows()