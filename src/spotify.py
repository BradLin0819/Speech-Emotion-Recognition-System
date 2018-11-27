import sys
import spotipy
import spotipy.util as util

username = 'han'
scope = 'user-library-read'
token=util.prompt_for_user_token(username, scope)

spotify = spotipy.Spotify(auth=token)

def get_category_playlists(category):
    mood_cat = spotify.category_playlists(category_id=category, country='tw', limit=50, offset=0)
    for playlist in mood_cat['playlists']['items']:
        name=playlist['name']
        playlist_id=playlist['id']
        print("Playlist Name=",name," id=",playlist_id)

sad_id='37i9dQZF1DWV3IJ2kse1M3'  #sad
anxiety_id='37i9dQZF1DX8gS5vh05dnc' #anxiety
hate_id='37i9dQZF1DX3YSRoSdA634' #Life Sucks
happy_id='37i9dQZF1DXdPec7aLTmlC' #Happy Hits!
angry_id='37i9dQZF1DX1s9knjP51Oa' #Calm Vibes
boring_id='37i9dQZF1DWWjGdmeTyeJ6' #Fresh Finds
neutral_id='37i9dQZF1DX7K31D69s4M1' #Piano in the Background
        

def get_recommended_songs(emotion):
    if emotion == 'sad':
        playlist=sad_id
    if emotion == 'anxiety':
        playlist=anxiety_id
    if emotion == 'hate':
        playlist=hate_id
    if emotion == 'happy':
        playlist=happy_id
    if emotion == 'angry':
        playlist=angry_id
    if emotion == 'boring':
        playlist=boring_id
    if emotion == 'neutral':
        playlist=neutral_id
    play=spotify.user_playlist(username,playlist_id=playlist,fields="tracks")
    recommended_songs=[]
    recommended_songs=get_tracks(play['tracks'])
    return recommended_songs


def get_tracks(tracks):
    songs=[]
    for i, item in enumerate(tracks['items']):
        track = item['track']
        song_url=track['preview_url']
        song_name=track['name']
        song_artist=track['artists'][0]['name']
        song_image_link=track['album']['images'][2]['url']
        song=[song_name,song_artist,song_image_link,song_url]
        songs.append(song)
    return songs


#get_category_playlists('mood')
if __name__ == '__main__':
    print(get_recommended_songs('sad'))
