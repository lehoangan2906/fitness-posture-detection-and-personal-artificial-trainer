import pygame
class textButton:
    def __init__(self, text, text_position,size,screen,color,hoverColor):
        self.text = text
        self.text_position = text_position
        self.size = size
        self.screen = screen
        self.color = color
        self.hoverColor = hoverColor
    
    def draw_text_button(self,mouse_position):
        font = pygame.font.SysFont('sans',self.size)
        self.text_render = font.render(self.text, True ,self.color)
        if (self.is_mouse_on_text(mouse_position)):
            self.text_render = font.render(self.text, True ,self.hoverColor)
            self.screen.blit(self.text_render,self.text_position)
            pygame.draw.line(self.screen,self.hoverColor,(self.text_position[0],self.text_position[1]+self.text_box[3]+1),(self.text_position[0]+self.text_box[2],self.text_position[1]+self.text_box[3]+1))
        else:
            self.screen.blit(self.text_render,self.text_position)
    
    def draw_text(self):
        font = pygame.font.SysFont('sans',self.size)
        self.text_render = font.render(self.text, True ,self.color)
        self.screen.blit(self.text_render,self.text_position)

    def is_mouse_on_text(self,mouse_position):
        self.text_box = self.text_render.get_rect()
        if mouse_position[0]>=self.text_position[0] and mouse_position[0]<=(self.text_position[0]+self.text_box[2]) and mouse_position[1]<=(self.text_position[1]+self.text_box[3]) and mouse_position[1]>=self.text_position[1]:
            return True
        return False