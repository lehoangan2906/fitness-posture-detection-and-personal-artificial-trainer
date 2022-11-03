
import pygame, sys
from pygame.locals import *
from inference import modelChoice
from inferenceSquat import squatDetect

from textButton import textButton as pen


screen_size = (1920,1080)
test_screen_size =(1800,900)
screen_demo_size = (576,320)
background_color =(150,150,150)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,200,0)
BLUE = (0,0,255)
YELLOW = (147,153,35)
AQUA = (0,200,200)
FUCHSIA = (255,0,255)
ORANGE = (255,125,25)
GRAPE =(100,25,125)
GRASS = (55,155,65)
WHITE = (255,255,255)




# init screen
pygame.init()
# screen_demo = pygame.display.set_mode(screen_demo_size)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("Gym Pose Correction")
clock = pygame.time.Clock()


bgMain = pygame.image.load('background\pexels-leon-ardho-1552242.png')
bgMain = pygame.transform.scale(bgMain, screen_size)

bgSquat = pygame.image.load('background\photo_2018-12-03_13-27-48_6452f067-f117-4be8-8584-2565e83bd17d_1024x1024.jpg')
bgSquat = pygame.transform.scale(bgSquat, screen_size)

bgPushUp = pygame.image.load('background\wallpapersden.com_chris-hemsworth-push-ups_1920x1080.jpg')
bgPushUp = pygame.transform.scale(bgPushUp, screen_size)

bgSitUP = pygame.image.load('background\\active-adult-athlete-situp-exercise-800x450.jpg')
bgSitUP = pygame.transform.scale(bgSitUP, screen_size)




def mainScreen(mouse_position,hello,pushup,squat,sitUp):
    screen.fill(background_color)
    screen.blit(bgMain,(0,0) )
    hello.draw_text()
    pushup.draw_text_button(mouse_position)
    squat.draw_text_button(mouse_position)
    sitUp.draw_text_button(mouse_position)
    

check = 0
def squatScreen(backMainScreen,mouse_position):
    global check
    screen.fill(background_color)
    # screen_demo.fill(RED)
    screen.blit(bgSquat,(0,0) ) 
    backMainScreen.draw_text_button(mouse_position)
    if check == 0:
        check = check + 1
    elif check ==1:
        squatDetect()
        check = check + 1
        
    
    

def pushupScreen(backMainScreen,mouse_position):
    global check
    screen.fill(background_color)
    screen.blit(bgPushUp,(0,0) )   
    backMainScreen.draw_text_button(mouse_position)
    if check == 0:
        check = check + 1
    elif check ==1:
        modelChoice(2)
        check = check + 1
    

def sitUpScreen(backMainScreen,mouse_position):
    global check
    screen.fill(background_color)
    screen.blit(bgSitUP,(0,0) )   
    backMainScreen.draw_text_button(mouse_position)  
    if check == 0:
        check = check + 1
    elif check ==1:
        modelChoice(3)
        check = check + 1

def main():
    global check
    numOfscreen = 0
    running = True
    hello = pen("Welcome",(600,100),200,screen,AQUA,BLUE)
    pushup = pen("Push Up",(700,400),120,screen,WHITE,BLUE)
    squat = pen("Squat",(700,600),120,screen,WHITE,BLUE)
    sitUp = pen("Sit Up",(700,800),120,screen,WHITE,BLUE)
    back = pen("Back",(250,150),40,screen,RED,BLUE)
    push = pen("Push Up",(600,100),200,screen,AQUA,BLUE)
    
    
    
    
    while running:
        clock.tick(30)
        mouse_position = pygame.mouse.get_pos()
        
        if(numOfscreen == 0):
            mainScreen(mouse_position,hello,pushup,squat,sitUp)
        elif numOfscreen == 1:
            squatScreen(back,mouse_position)
        elif numOfscreen ==2:
            pushupScreen(back,mouse_position)
        else: 
            sitUpScreen(back,mouse_position)

        
        
        
        for event in pygame.event.get():
            #squat
            if event.type== pygame.MOUSEBUTTONDOWN:
                if squat.is_mouse_on_text(mouse_position)== True and event.button == 1:
                    numOfscreen = 1
                    check = 0
                    
                elif pushup.is_mouse_on_text(mouse_position)== True and event.button == 1:
                    numOfscreen = 2
                    check = 0
                    
                elif sitUp.is_mouse_on_text(mouse_position)== True and event.button == 1:
                    numOfscreen = 3
                    check = 0
                    
                elif numOfscreen!=0 and back.is_mouse_on_text(mouse_position)== True and event.button == 1:
                    numOfscreen = 0
            #quit    
            if event.type == QUIT or(event.type == KEYDOWN and event.key == K_ESCAPE ):
                pygame.quit()
                sys.exit()

        
        pygame.display.update()
        pygame.display.flip()
        

main()

