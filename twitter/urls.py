from django.urls import path
from twitter import views

app_name = 'twitter'

urlpatterns = [
    path("",views.test,name='test'),
    path("testing/",views.read_root,name='read_root'),
    path("info/<str:user_name>",views.info,name='info'),
    path("following/<int:user_id>",views.following,name='following'),
    path("tweets/<str:user_name>",views.tweets,name='tweets'),
    path("sentiments/<str:user_name>",views.sentiments,name='sentiments'),
    path("wordcloud/<str:user_name>",views.wordcloud,name='wordcloud'),
]