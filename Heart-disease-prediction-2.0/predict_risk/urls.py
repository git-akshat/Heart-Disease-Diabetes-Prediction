from django.conf.urls import url
from . import views
app_name='predict'
urlpatterns=[
url(r'^(?P<pk>\d+)$',views.heart,name='predict'),
url(r'^description/(?P<pk>\d+)$',views.description,name='description'),
url(r'^diabetes/(?P<pk>\d+)$', views.diabetes,name='diabetes'),
url(r'^heart/(?P<pk>\d+)$', views.heart,name='heart'),
]
