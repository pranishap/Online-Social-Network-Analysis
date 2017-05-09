# coding: utf-8


from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI

consumer_key = 'EJJF4oaMdma5Db3CUFDF7ojGQ'
consumer_secret = 'EaTt1J215ahsdsr0GKiYnCyHxFCfpaU5L7jB45Lz1sx2qie0JM'
access_token = '139760169-MyvKFFfkzQol1Rza3aMorE9pkxqxZCcUluEgEDaS'
access_token_secret = 'FKvBpyRDsNadw7tVnEc5ZYBty6QiYFyN4zSMHZQfdZ1wM'


def get_twitter():

    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def read_screen_names(filename):

    with open(filename, "r") as ins:
        array = []
        for line in ins:
            array.append(line.rstrip('\n'))
    ins.closed
    return array



def robust_request(twitter, resource, params, max_tries=5):

    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


def get_users(twitter, screen_names):
    request = robust_request(twitter, 'users/lookup', {'screen_name': screen_names}, max_tries=5)
    users = [r for r in request]
    userList = list(users)
    return userList



def get_friends(twitter, screen_name):
    request = robust_request(twitter, 'friends/ids', {'screen_name': screen_name, 'count':5000}, max_tries=5)
    friends = [r for r in request]
    friendsList = list(friends)
    return sorted(friendsList)


def add_all_friends(twitter, users):
    for user in users:
        screen_name = user['screen_name']
        user['friends'] = get_friends(twitter, screen_name)


def print_num_friends(users):
    users = sorted(users, key=lambda x: x['screen_name'])
    for user in users:
        screen_name = user['screen_name']
        friends = user['friends']
        print(screen_name + ":" + str(len(friends)))



def count_friends(users):
    c = Counter()
    for user in users:
        friends = user['friends']
        c.update(friends)
    return c


def friend_overlap(users):
    overlapTuple = []
    for i in range(len(users)):
        for j in range(i+1,len(users)):
            if i!=j:
                seti = set(users[i]['friends'])
                setj = set(users[j]['friends'])
                finalSet = seti & setj
                overlapString = users[i]['screen_name'] , users[j]['screen_name'] ,len(finalSet)
                overlapTuple.append(overlapString)
    return sorted(overlapTuple, key=lambda x: -x[2])




def followed_by_hillary_and_donald(users, twitter):
    list = []
    for j in range(len(users)):
        if (users[j]['screen_name'] == 'HillaryClinton' or users[j]['screen_name'] == 'realDonaldTrump'):
            list.append(users[j])

    setOne = set(list[0]['friends'])
    setTwo = set(list[1]['friends'])
    commonFriend = setOne & setTwo
    request = robust_request(twitter, 'users/lookup', {'user_id': commonFriend}, max_tries=5)
    cFriend = [r for r in request]
    name = cFriend[0]['screen_name']
    return name


def create_graph(users, friend_counts):
    graph = nx.Graph()
    list = []
    for i in friend_counts:
        if (friend_counts[i] >= 2):
            graph.add_node(i)
            list.append(i)
    for user in users:
        graph.add_node(user['screen_name'])
        friends = user['friends']
        for friend in friends:
            if (friend in list):
                graph.add_edge(friend,user['screen_name'], weight=(friend_counts[friend]*10))
    return graph




def draw_network(graph, users, filename):
    labels = {}
    for user in users:
        labels[user['screen_name']] = user['screen_name']
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)
    ax.set_title('Graph - Shapes', fontsize=12)
    pos = nx.spring_layout(graph, dim=2, k=0.15, pos=None, fixed=None, iterations=50, weight='weight', scale=1.0, center= None)
    nx.draw(graph, pos, node_size=80,node_color='#f08080',edge_color='#c5c5c5',labels = labels,with_labels = True, font_size=10, font_weight='bold')
    plt.tight_layout()
    plt.savefig("network.png", format="PNG")
    plt.show()


def main():
    twitter = get_twitter()
    screen_names = read_screen_names('candidates.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    add_all_friends(twitter, users)
    print('Friends per candidate:')
    print_num_friends(users)
    friend_counts = count_friends(users)
    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))
    print('Friend Overlap:\n%s' % str(friend_overlap(users)))
    print('User followed by Hillary and Donald: %s' % followed_by_hillary_and_donald(users, twitter))

    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network(graph, users, 'network.png')
    print('network drawn to network.png')


if __name__ == '__main__':
    main()
