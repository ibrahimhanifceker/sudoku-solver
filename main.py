from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image
from collections import deque

def solve(row, col):
    if row == 9:
        global solution
        solution = [[arr[i][j] for j in range(9)] for i in range(9)]
        print("solution:")
        for i in arr:
            print(i)
        return
    if arr[row][col] != 0:
        solve(row + (col + 1) // 9, (col + 1) % 9)
        return
    for i in range(1, 10):
        if rows[row][i] == 0 and cols[col][i] == 0 and sqrs[(row // 3) * 3 + (col // 3)][i] == 0:
            rows[row][i] = 1
            cols[col][i] = 1
            sqrs[(row // 3) * 3 + (col // 3)][i] = 1
            arr[row][col] = i
            solve(row + (col + 1) // 9, (col + 1) % 9)
            rows[row][i] = 0
            cols[col][i] = 0
            sqrs[(row // 3) * 3 + (col // 3)][i] = 0
            arr[row][col] = 0
    return

def getColor(pixel):
    if pixel < 100:
        return 1
    return 0

def isValid(x, y):
    if x < 0 or x >= height or y < 0 or y >= width:
        return False
    if vis[x][y] == 1:
        return False
    if getColor(image[x, y]) == 0:
        return False
    return True

q = deque()

def bfs(x, y, typ):
    global mnX
    mnX = 1e9
    global mxX
    mxX = -1e9
    global mnY
    mnY = 1e9
    global mxY
    mxY = -1e9
    global cnt2
    cnt2 = 0
    q.append([x, y])
    while len(q) != 0:
        point = q.popleft()
        if vis[point[0]][point[1]] == 1:
            continue
        vis[point[0]][point[1]] = 1
        cnt2 += 1
        if typ == 1:
            rowcnt[point[0]] += 1
            colcnt[point[1]] += 1
        mnX = min(mnX, point[0])
        mxX = max(mxX, point[0])
        mnY = min(mnY, point[1])
        mxY = max(mxY, point[1])
        for X in range(-1, 2):
            for Y in range(-1, 2):
                if abs(X + Y) == 1 and isValid(point[0] + X, point[1] + Y):
                    q.append([point[0] + X, point[1] + Y])
    return


app = FastAPI()

@app.post("/")
async def index(file: UploadFile = File(...)):
    with open(f"uploads/{file.filename}", "wb") as buffer:
        buffer.write(await file.read())

    global arr
    arr = [[0] * 9 for _ in range(9)]
    global rows
    rows = [[0] * 10 for _ in range(9)]
    global cols
    cols = [[0] * 10 for _ in range(9)]
    global sqrs
    sqrs = [[0] * 10 for _ in range(9)]

    global image

    image = np.array(cv2.imread(f"uploads/{file.filename}"))

    if image is None:
        print("error")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("deneme", image)
    #cv2.waitKey(0)

    global height
    height = image.shape[0]
    global width
    width = image.shape[1]

    if width * height > 360000:
        width //= 2
        height //= 2
        image = cv2.resize(image, dsize = (width, height), interpolation=cv2.INTER_CUBIC)

    global vis
    vis = [[0] * width for _ in range(height)]

    global rowcnt
    rowcnt = [0] * height
    global colcnt
    colcnt = [0] * width

    cnt = 0

    yes = 1

    horizontal = []
    vertical = []

    for i in range(height):
        for j in range(width):
            #print(i, j)
            if vis[i][j] == 0 and getColor(image[i, j]) == 1:
                #print(i, "aaa", j)
                #print(image[i, j])
                if yes == 1:
                    bfs(i, j, 1)
                    for k in range(10):
                        horizontal.append(mnX + (mxX - mnX) / 9 * k)
                    for k in range(10):
                        vertical.append(mnY + (mxY - mnY) / 9 * k)
                    yes = 0
                    continue
                bfs(i, j, 0)
                if cnt2 < 15:
                    continue
                cnt += 1
                lenX = mxX - mnX + 1
                lenY = mxY - mnY + 1
                mnScore = 100000000000
                digit = -1
                for k in range(1, 10):
                    for l in range(1, 3):
                        img = cv2.imread('numbers/number_' + str(k) + '/number_' + str(k) + '_' + str(l) + '.jpg')
                        if img is None:
                            print("error")
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = cv2.resize(img, dsize = (lenY, lenX), interpolation=cv2.INTER_CUBIC)
                        #cv2.imshow("Image Window", img)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()
                        score = 0
                        image = image.astype(np.int32)
                        img = img.astype(np.int32)
                        for x in range(lenX):
                            for y in range(lenY):
                                if getColor(image[x + mnX, y + mnY]) == getColor(img[x, y]):
                                    score += 2
                                else:
                                    score -= 3
                                #score += (0 + abs(image[x + mnX, y + mnY][0] - img[x, y][0]) + abs(image[x + mnX, y + mnY][1] - img[x, y][1]) + abs(image[x + mnX, y + mnY][2] - img[x, y][2])) ** 2
                        score = -score
                        if score < mnScore:
                            mnScore = score
                            digit = k
                #print(digit, mnScore)
                rowind = -1
                colind = -1
                for k in range(len(horizontal) - 1):
                   if mnX >= horizontal[k] and mnX <= horizontal[k + 1]:
                       rowind = k
                for k in range(len(vertical) - 1):
                   if mnY >= vertical[k] and mnY <= vertical[k + 1]:
                       colind = k
                #print(rowind, colind)
                arr[rowind][colind] = digit

    
    for i in range(9):
        for j in range(9):
            if arr[i][j] != 0:
                rows[i][arr[i][j]] = 1
                cols[j][arr[i][j]] = 1
                sqrs[(i // 3) * 3 + j // 3][arr[i][j]] = 1


    for i in arr:
        print(i)
        
    solve(0, 0)
    
    response = dict()

    for i in range(9):
        response[str(i)] = dict()
        for j in range(9):
            response[str(i)][str(j)] = solution[i][j]
    
    return response
