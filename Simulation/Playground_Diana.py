import csv
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#read orders from csv file into a list of dictionaries
with open ('Playground_Diana.csv','r') as file:
    csv_reader=csv.DictReader(file)
    data=[row for row in csv_reader]

# for obj in data:
#     print(obj)

#make table, keys are the headers
#each row represents an order
fig = go.Figure(data=[go.Table(header=dict(values=[key for key in data[0].keys()]),
                 cells=dict(values=[[x['order_id'] for x in data], [x['objects'] for x in data],[x['origin'] for x in data],
                                   [x['destination'] for x in data],[x['interval'] for x in data]]))
                 ])

fig.update_layout(margin={"r": 0, "t": 200, "l": 800, "b": 0})

fig.show()