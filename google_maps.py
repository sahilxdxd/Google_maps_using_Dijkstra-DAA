import numpy as np
import matplotlib.pyplot as plt
import heapq
import random

# ---------------------------
# Step 1: Map Configuration
# ---------------------------
rows, cols = 10, 10
obstacle_ratio = 0.2

grid = np.zeros((rows, cols), dtype=int)
num_obstacles = int(rows * cols * obstacle_ratio)
for _ in range(num_obstacles):
    r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
    grid[r, c] = -1

# ---------------------------
# Step 2: Interactive Colored Point Selection with Labels
# ---------------------------
plt.figure(figsize=(6, 6))
plt.imshow(grid == -1, cmap='gray', alpha=1.0)
plt.title("Click: Start (Blue) → Checkpoints (Orange) → Goal (Gold)\nPress Enter to finish", fontsize=10)
plt.xticks(np.arange(cols))
plt.yticks(np.arange(rows))
plt.gca().invert_yaxis()

points = []
colors = {'start': 'blue', 'checkpoint': 'orange', 'goal': 'gold'}

while True:
    pts = plt.ginput(1, timeout=-1)
    if not pts:
        break
    x, y = pts[0]
    r, c = int(round(y)), int(round(x))
    points.append((r, c))

    # Determine point type
    if len(points) == 1:
        label = "Start"
        color = colors['start']
    else:
        label = f"Checkpoint {len(points)-1}"
        color = colors['checkpoint']

    # Plot small dot + label
    plt.scatter(c, r, color=color, s=60, marker='o')
    plt.text(c + 0.2, r, label, color=color, fontsize=9, fontweight='bold')
    plt.draw()

# Convert last to goal if at least 2 points selected
if len(points) >= 2:
    goal = points[-1]
    plt.scatter(goal[1], goal[0], color=colors['goal'], s=80, marker='*')
    plt.text(goal[1] + 0.2, goal[0], "Goal", color=colors['goal'], fontsize=9, fontweight='bold')
    plt.draw()

plt.close()

if len(points) < 2:
    print("Please select at least a start and goal point.")
    exit()

start = points[0]
goal = points[-1]
checkpoints = points[1:-1]

grid[start] = 0
grid[goal] = 0
for cp in checkpoints:
    grid[cp] = 0

# ---------------------------
# Step 3: Dijkstra Algorithm
# ---------------------------
def dijkstra(grid, start, goal):
    rows, cols = grid.shape
    dist = np.full((rows, cols), np.inf)
    prev = np.full((rows, cols, 2), -1, dtype=int)
    visited = set()
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        d, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            break
        r, c = current
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] != -1:
                new_dist = d + 1
                if new_dist < dist[nr, nc]:
                    dist[nr, nc] = new_dist
                    prev[nr, nc] = [r, c]
                    heapq.heappush(pq, (new_dist, (nr, nc)))
    return dist, prev, visited

def reconstruct_path(prev, start, goal):
    path = []
    current = goal
    while tuple(current) != start and np.all(current != [-1, -1]):
        path.append(tuple(current))
        current = prev[tuple(current)]
    path.append(start)
    path.reverse()
    return path

# ---------------------------
# Step 4: Handle Checkpoints Sequentially
# ---------------------------
full_path = []
total_visited = set()
total_distance = 0
points_seq = [start] + checkpoints + [goal]

for i in range(len(points_seq)-1):
    s, g = points_seq[i], points_seq[i+1]
    dist, prev, visited = dijkstra(grid, s, g)
    segment_path = reconstruct_path(prev, s, g)
    if np.isinf(dist[g]):
        print(f"\n⚠️ No path found between {s} and {g}")
        continue
    total_visited |= visited
    total_distance += int(dist[g])
    if full_path and segment_path[0] == full_path[-1]:
        full_path += segment_path[1:]
    else:
        full_path += segment_path

# ---------------------------
# Step 5: Visualization
# ---------------------------
plt.figure(figsize=(7, 7))
plt.imshow(grid == -1, cmap='gray', alpha=1.0)
plt.xticks(np.arange(cols))
plt.yticks(np.arange(rows))
plt.gca().invert_yaxis()
plt.gca().set_facecolor('white')

# Explored nodes
for (r, c) in total_visited:
    if (r, c) not in full_path and grid[r, c] != -1:
        plt.scatter(c, r, color='lightgreen', s=25, alpha=0.6)

# Path
if len(full_path) > 1:
    y_coords = [r for r, c in full_path]
    x_coords = [c for r, c in full_path]
    plt.plot(x_coords, y_coords, color='red', linewidth=2.5, label='Shortest Path')

# Markers + Labels
plt.scatter(start[1], start[0], color='blue', s=100, marker='o', label='Start')
plt.text(start[1]+0.2, start[0], "Start", color='blue', fontsize=9, fontweight='bold')

plt.scatter(goal[1], goal[0], color='gold', s=120, marker='*', label='Goal')
plt.text(goal[1]+0.2, goal[0], "Goal", color='gold', fontsize=9, fontweight='bold')

for i, cp in enumerate(checkpoints, start=1):
    plt.scatter(cp[1], cp[0], color='orange', s=80, marker='D', label=f'Checkpoint {i}')
    plt.text(cp[1]+0.2, cp[0], f"C{i}", color='orange', fontsize=9, fontweight='bold')

plt.legend(loc='upper right', fontsize=8)
plt.title("Dynamic Dijkstra Path with Labeled Checkpoints", fontsize=12)
plt.show()

# ---------------------------
# Step 6: Output Path Info
# ---------------------------
if len(full_path) < 2:
    print("\n❌ No valid path found.")
else:
    print("\n✅ Shortest Path (with checkpoints):")
    for i, (r, c) in enumerate(full_path, start=1):
        print(f"Step {i:2d}: ({r}, {c})")
    print("\nTotal Distance:", total_distance)
