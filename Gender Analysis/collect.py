"""
collect.py
"""
import requests
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import pickle
import urllib.request


consumer_key = 'EJJF4oaMdma5Db3CUFDF7ojGQ'
consumer_secret = 'EaTt1J215ahsdsr0GKiYnCyHxFCfpaU5L7jB45Lz1sx2qie0JM'
access_token = '139760169-MyvKFFfkzQol1Rza3aMorE9pkxqxZCcUluEgEDaS'
access_token_secret = 'FKvBpyRDsNadw7tVnEc5ZYBty6QiYFyN4zSMHZQfdZ1wM'

"""consumer_key = 'ulHlYA2RlUeH6otzaqGRj6w8J'
consumer_secret = '2P1w5xUNCEt9NsP6ed8gIfBzGUz51VfYSGJMBltIDc1pId2fGU'
access_token = '46917119-NNoPlOc83ZKskKoxTstpOENYXwhcfxhMEtdlT8oz3'
access_token_secret = 'CSF8ap3Fah1p4z0ymgPggxwoFBWn8PBk4LJxf9iOdArVw'"""


all_League = ['Orioles','RedSox','whitesox','Indians','tigers','astros','Royals','Angels','Twins','Yankees','Athletics','Mariners','RaysBaseball','Rangers','BlueJays','Dbacks','Braves','Cubs','Reds','Rockies','Dodgers','Marlins','Brewers','Mets','Phillies','Pirates','Padres','SFGiants','Cardinals','Nationals']

national_League = ['Arizona Diamondbacks','Atlanta Braves','Chicago Cubs','Cincinnati Reds','Colorado Rockies','Los Angeles Dodgers','Miami Marlins','Milwaukee Brewers','New York Mets','Philadelphia Phillies','Pittsburgh Pirates','San Diego Padres','San Francisco Giants''St. Louis Cardinals','Washington Nationals']

def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def robust_request(twitter, resource, params, max_tries=5):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def intersect(*d):
    sets = iter(map(set, d))
    result = sets.next()
    for s in sets:
        result = result.intersection(s)
    return result


def get_followers(twitter,screen_name):
    followerDetails = {}
    request = robust_request(twitter,'followers/list', {'screen_name': screen_name,'count': 200})
    followers = []
    for r in request:
        length = len(r['description'])
        if(length > 1):
            followers.append(r['screen_name'])
    followersList = list(followers)
    for r in request:
        length = len(r['description'])
        if(length > 1):
            dict = {}
            dict['description'] = r['description']
            dict['name'] = r['name']
            dict['screen_name'] = r['screen_name']
            followerDetails[r['screen_name']] = dict
    return followersList,followerDetails

def get_friends_of_common_followers(twitter,screen_names):
    friendsList = {}
    friendsL = []
    for screen_name in screen_names:
        request = robust_request(twitter,'friends/list', {'screen_name': screen_name,'count': 200})
        friends = [r['screen_name'] for r in request]
        leagueFollowers = list(set(list(friends)) & set(all_League))
        friendsList[screen_name] = leagueFollowers
        friendsL.append(leagueFollowers)
    return friendsList



def add_all_followers(twitter):
    u =[]
    followerDetails ={}
    screen_names=[]
    screen_names.append('Cubs')
    screen_names.append('Indians')
    screen_names.append('whitesox')
    screen_names.append('Yankees')
    screen_names.append('SFGiants')
    request = robust_request(twitter, 'users/lookup', {'screen_name': screen_names}, max_tries=5)
    users = [r for r in request]
    userList = list(users)
    for user in userList:
        screen_name = user['screen_name']
        user['followers'],followerDet = get_followers(twitter, screen_name)
        followerDetails.update(followerDet)
    cubs_followers_set = set(userList[0]['followers'])
    indian_followers_set = set(userList[1]['followers'])
    whitesox_set = set(userList[2]['followers'])
    yankees_set = set(userList[3]['followers'])
    sfgiants_set = set(userList[4]['followers'])

    friendsList = []
    common_followers = cubs_followers_set & indian_followers_set
    friendsList.extend(list(cubs_followers_set & whitesox_set))
    friendsList.extend(list(cubs_followers_set & yankees_set))
    friendsList.extend(list(cubs_followers_set & sfgiants_set))
    friendsList.extend(list(indian_followers_set & whitesox_set))
    friendsList.extend(list(indian_followers_set & yankees_set))
    friendsList.extend(list(indian_followers_set & sfgiants_set))
    friendsList.extend(list(whitesox_set & yankees_set))
    friendsList.extend(list(whitesox_set & sfgiants_set))
    friendsList.extend(list(yankees_set & sfgiants_set))
    friendListSet = set(friendsList)
    return userList,friendListSet,followerDetails


def get_census_names():
    males_url = 'http://www2.census.gov/topics/genealogy/' + \
    '1990surnames/dist.male.first'
    females_url = 'http://www2.census.gov/topics/genealogy/' + \
    '1990surnames/dist.female.first'
    males = requests.get(males_url).text.split('\n')
    females = requests.get(females_url).text.split('\n')
    males_pct = get_percents(males)
    females_pct = get_percents(females)
    male_names = set([m.split()[0].lower() for m in males if m])
    female_names = set([f.split()[0].lower() for f in females if f])
    
    male_names = set([m for m in male_names if m not in female_names or
                      males_pct[m] > females_pct[m]])
    female_names = set([f for f in female_names if f not in male_names or
                                          females_pct[f] > males_pct[f]])
    return male_names,female_names


def get_percents(name_list):
    return dict([(n.split()[0].lower(), float(n.split()[1]))
                 for n in name_list if n])



def main():
    twitter = get_twitter()
    users,friendsList,followerDetails = add_all_followers(twitter)
    male_names,female_names = get_census_names()
    pickle.dump(users, open('users.pkl', 'wb'))
    pickle.dump(friendsList, open('friendList.pkl', 'wb'))
    pickle.dump(followerDetails, open('followerDetails.pkl', 'wb'))
    pickle.dump(male_names, open('maleNames.pkl', 'wb'))
    pickle.dump(female_names, open('femaleNames.pkl', 'wb'))



if __name__ == '__main__':
    main()
