import pygame
import pygame.camera
import os
import random

cam_port = 0

WIN = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
WIDTH, HEIGHT = pygame.display.Info().current_w, pygame.display.Info().current_h

pygame.display.set_caption("Data Collector")
FPS = 30

INTRO = pygame.image.load(os.path.join('assets', 'intro.png'))
END = pygame.image.load(os.path.join('assets', 'end.png'))
DOT = pygame.image.load((os.path.join('assets', 'dot.png')))

DURATION = 3

main_dir = "Data/raw"
locations_dir = ["HG", "HM", "HD", "MG", "MM", "MD", "BG", "BM", "BD"]


def intro_window():
    WIN.fill((0, 0, 0))
    WIN.blit(INTRO, ((WIDTH - INTRO.get_width()) / 2, (HEIGHT - INTRO.get_height()) / 2))

    pygame.display.update()


def end_window():
    WIN.fill((0, 0, 0))
    # WIN.blit(CAM.get_image(), (0, 0))
    WIN.blit(END, ((WIDTH - END.get_width()) / 2, (HEIGHT - END.get_height()) / 2))

    pygame.display.update()


def main_window(x, y):
    WIN.fill((0, 0, 0))
    WIN.blit(DOT, (x - DOT.get_width() / 2, y - DOT.get_height() / 2))

    pygame.display.update()


def start_camera():
    pygame.camera.init()
    camlist = pygame.camera.list_cameras()
    CAM = pygame.camera.Camera(camlist[cam_port], (640, 480))
    CAM.start()
    return CAM


def position(location):
    screen_x = WIDTH // 3
    screen_y = HEIGHT // 3

    x = random.randrange(screen_x * (location % 3) + DOT.get_width() / 2,
                         screen_x * (location % 3 + 1) - DOT.get_width() / 2)
    y = random.randrange(screen_y * (location // 3) + DOT.get_height() / 2,
                         screen_y * (location // 3 + 1) - DOT.get_height() / 2)
    return x, y, location


def take_picture(CAM):
    img = CAM.get_image()
    return img


def save_picture(location, img):
    id_ = str(random.randrange(1000000000))
    loc = main_dir + "/" + locations_dir[location] + "/img" + id_ + ".png"

    pygame.image.save(img, loc)


def main():
    STEP = "intro"
    LEN = DURATION * 9
    locations = DURATION * list(range(9))
    random.shuffle(locations)
    random_x, random_y = 0, 0
    CAM = None

    sec = 3
    tick = -1
    delta_sec = .5  # Time needed for the user to actualize his eye position to the next dot

    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        if STEP == "intro":
            intro_window()

        elif STEP == "main":
            if LEN <= 0:
                STEP = "end"
            else:
                if tick <= 0:
                    LEN -= 1
                    tick = FPS * sec
                    random_x, random_y, location = position(locations[LEN])
                else:
                    tick -= 1

                    if tick <= FPS * int(sec - delta_sec) and not tick % 3:
                        img = take_picture(CAM)
                        save_picture(locations[LEN], img)

                main_window(random_x, random_y)

        elif STEP == "end":
            end_window()

        keys_pressed = pygame.key.get_pressed()
        if keys_pressed[pygame.K_SPACE]:
            if STEP == "intro":
                STEP = "main"
                CAM = start_camera()

            elif STEP == "end":
                run = False

        if keys_pressed[pygame.K_RETURN] and STEP == "end":
            STEP = "main"
            LEN = DURATION * 9

    pygame.quit()


if __name__ == "__main__":
    main()




