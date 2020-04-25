from django.conf.urls import url,include
from . import views
app_name = 'accounts'
urlpatterns=[
    url(r'^register/$', views.register, name='register'),
    url(r'^logout/$',views.user_logout,name='logout'),
    url(r'^profile/(?P<pk>\d+)/$', views.ProfileDetailView.as_view(), name='profile'),
    url(r'^predict/',include('predict_risk.urls', namespace="predict")),
#url(r'^profile/(?P<pk>\d+)/edit/$', views.profile_update, name='edit_profile'),
]
