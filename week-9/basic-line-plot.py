import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 40]
plt.plot(x, y, marker='o')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Basic Line Plot')

# Save the plot as an image file locally
plt.savefig('line_plot.png', dpi=300, bbox_inches='tight') 
plt.show()
