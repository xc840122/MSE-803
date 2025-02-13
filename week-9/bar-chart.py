import matplotlib.pyplot as plt
categories = ['A', 'B', 'C', 'D']
values = [3, 7, 1, 8]
plt.bar(categories, values, color='skyblue')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart Example')

# Save the plot as an image file locally
plt.savefig('bar_chart.png', dpi=300, bbox_inches='tight')
plt.show()
