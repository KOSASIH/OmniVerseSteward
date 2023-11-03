```python
import pandas as pd
import matplotlib.pyplot as plt

# Load historical data on multiverse disturbances
disturbances_data = pd.read_csv('multiverse_disturbances.csv')

# Analyze recurring patterns of disturbances
disturbances_patterns = disturbances_data.groupby(['Dimension', 'Year']).size().unstack(fill_value=0)

# Generate line plot to visualize long-term trends
disturbances_patterns.plot(kind='line', marker='o')
plt.xlabel('Year')
plt.ylabel('Number of Disturbances')
plt.title('Long-Term Trends of Multiverse Disturbances')
plt.legend(title='Dimension')
plt.show()

# Interactive features for exploring and querying the data
def query_disturbances_data(start_year, end_year, dimension=None):
    if dimension:
        filtered_data = disturbances_data[(disturbances_data['Year'] >= start_year) & 
                                          (disturbances_data['Year'] <= end_year) & 
                                          (disturbances_data['Dimension'] == dimension)]
    else:
        filtered_data = disturbances_data[(disturbances_data['Year'] >= start_year) & 
                                          (disturbances_data['Year'] <= end_year)]
    
    return filtered_data

# Example usage of the query_disturbances_data function
start_year = 2000
end_year = 2020
dimension = 'Dimension X'
filtered_data = query_disturbances_data(start_year, end_year, dimension)
print(filtered_data)
```

This code provides a basic implementation of an AI-powered data analysis and visualization tool for multiverse disturbances. 

The code assumes that historical data on multiverse disturbances is available in a CSV file named 'multiverse_disturbances.csv'. You may need to modify the code to match the actual format and structure of your data.

The tool first loads the historical data into a pandas DataFrame. It then analyzes the recurring patterns of disturbances by grouping the data by dimension and year. The resulting patterns are visualized using a line plot, where each line represents a dimension and the x-axis represents the years.

The tool also includes interactive features for exploring and querying the data. The `query_disturbances_data` function allows you to filter the data based on the start and end years, as well as the dimension. The function returns a filtered DataFrame that contains the relevant data. In the example usage provided, the function is used to filter the data for the years 2000 to 2020 and the dimension 'Dimension X'.

Feel free to customize and expand upon this code to meet your specific requirements and data structure.
