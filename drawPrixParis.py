from __future__ import print_function
import pandas as pd
from geopy.geocoders import Nominatim

import bokeh
from bokeh.io import output_file, show


from bokeh.models import (GMapPlot, GMapOptions, ColumnDataSource, Circle,
        DataRange1d, Range1d, PanTool, WheelZoomTool, BoxSelectTool,
        #PreviewSaveTool
        )
from bokeh.resources import INLINE
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.charts import Scatter

def getLoLa(addr):
    geolocator = Nominatim()
    location = geolocator.geocode(addr)
    la = location.latitude
    lo = location.longitude
    return lo, la

PrixParis = pd.read_csv('prixParis.csv')
PrixParis['Lo'], PrixParis['La'] = \
    zip(*PrixParis['Location'].map(getLoLa))


prixmax = PrixParis['Prix'].max()
prixmin = PrixParis['Prix'].min()
def getHexColor(prix) :
    prixx = int(prix*256./prixmax)
    col = '#%02x%2x%02x' % (prixx,50,150)

    print ('prix: ', prixx, 'color: ', col)
    return (col)
PrixParis['Color'] = PrixParis['Prix'].apply(getHexColor)


#output_notebook(resources=INLINE)
# JSON style string taken from: https://snazzymaps.com/style/1/pale-dawn
map_options = GMapOptions(lat=48.8003394, lng=2.2508597, map_type="roadmap", zoom=11, styles="""[{"featureType":"administrative","elementType":"all","stylers":[{"visibility":"on"},{"lightness":33}]},{"featureType":"landscape","elementType":"all","stylers":[{"color":"#f2e5d4"}]},{"featureType":"poi.park","elementType":"geometry","stylers":[{"color":"#c5dac6"}]},{"featureType":"poi.park","elementType":"labels","stylers":[{"visibility":"on"},{"lightness":20}]},{"featureType":"road","elementType":"all","stylers":[{"lightness":20}]},{"featureType":"road.highway","elementType":"geometry","stylers":[{"color":"#c5c6c6"}]},{"featureType":"road.arterial","elementType":"geometry","stylers":[{"color":"#e4d7c6"}]},{"featureType":"road.local","elementType":"geometry","stylers":[{"color":"#fbfaf7"}]},{"featureType":"water","elementType":"all","stylers":[{"visibility":"on"},{"color":"#acbcc9"}]}]
""")

# Google Maps now requires an API key. You can find out how to get one here:
# https://developers.google.com/maps/documentation/javascript/get-api-key
API_KEY = ""

x_range = DataRange1d()
y_range = DataRange1d()

plot = figure()

plot = GMapPlot(
    x_range=x_range, y_range=y_range,
    map_options=map_options,
    api_key=API_KEY,
)

source = ColumnDataSource(
    data=dict(
        lat=PrixParis['La'].tolist(),
        lon=PrixParis['Lo'].tolist(),
        cols=PrixParis['Color'],
    )
)

circle = Circle(x="lon", y="lat", size=15, fill_color="cols", fill_alpha=0.6,
        line_color='#FFFFFF',line_width=2)
plot.add_glyph(source, circle)

plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())

output_file("PrixParis.html")
show(plot)
