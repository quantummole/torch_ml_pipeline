import cv2
import numpy as np

def extract_grid_simple(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grid_img = 1.*(img.sum(axis=-1) < 10)
    m, n = grid_img.shape
    horizontals = np.nonzero(grid_img[:,n//2])[0]
    horizontal_unique = [horizontals[0]]
    for y in horizontals:
        if np.abs(horizontal_unique[-1] - y) > 5:
            horizontal_unique.append(y)
    verticals = np.nonzero(grid_img[m//2,:])[0]
    vertical_unique = [verticals[0]]
    for y in verticals:
        if np.abs(vertical_unique[-1] - y) > 5:
            vertical_unique.append(y)
    assert len(horizontal_unique) > 1, "horizontal lines not detected"
    assert len(vertical_unique) > 1, "vertical lines not detected"
    assert len(horizontal_unique) == len(vertical_unique), "grid not detected"
    queen_grid = np.zeros((len(horizontal_unique)-1,len(vertical_unique)-1))
    colors = {}
    for i,(x1,x2) in enumerate(zip(vertical_unique,vertical_unique[1:])):
        for j, (y1,y2) in enumerate(zip(horizontal_unique, horizontal_unique[1:])):
            median_x = (x1+x2)//2
            median_y = (y1+y2)//2
            color = tuple(img[median_y,median_x].tolist())
            idx = colors.get(color,len(colors))
            colors[color] = idx
            assert idx < queen_grid.shape[1], "color extraction failed"
            queen_grid[j,i] = idx
    return queen_grid, horizontal_unique, vertical_unique, colors


