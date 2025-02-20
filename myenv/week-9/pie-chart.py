import matplotlib.pyplot as plt

# Data for the pie chart
categories = ['A', 'B', 'C', 'D']
values = [3, 7, 1, 8]

# Create a pie chart
plt.pie(values, labels=categories, autopct='%1.1f%%', startangle=90, colors=['orange', 'blue', 'green', 'red'])
plt.title('Pie Chart Example')

# Save the plot as an image file locally
plt.savefig('pie_chart.png', dpi=300, bbox_inches='tight')

plt.show()