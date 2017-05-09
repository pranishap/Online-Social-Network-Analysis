"""
sumarize.py
"""
import pickle
import networkx as nx

def count_users(users,friendsList,followerDetails,graph):
    num_users = 0
    num_users = len(users)
    num_users = num_users + len(followerDetails.keys())
    components = list(nx.connected_component_subgraphs(graph))
    num_component = len(components)
    num_nodes = 0
    for component in components:
        num_nodes = num_nodes + nx.number_of_nodes(component)
    avg_nodes = num_nodes / num_component
    male_count = 0
    female_count = 0
    count = 0
    male_instance = ''
    female_instance = ''
    userNames = list(followerDetails.keys())
    for follower_username in userNames:
        followerDet = followerDetails[follower_username]
        if('gender' in followerDet.keys()):
            if (followerDet['gender'] == 'male'):
                male_count = male_count + 1
                if(len(male_instance) == 0):
                    male_instance = followerDet
            elif(followerDet['gender'] == 'female'):
                female_count = female_count + 1
                if(len(female_instance) == 0):
                    female_instance = followerDet
        else:
                    count = count + 1
    s = 'Number of users collected: ' + str(num_users) + '\n' + 'Number of communities discovered: ' + str(num_component) + '\n' + 'Average number of users per community: ' + str(avg_nodes) +'\n' + 'Number of instances per class found: \n\t Male : ' + str(male_count) + '\n'+ '\t Female : ' + str(female_count) + '\n' + 'Example of Male class : ' + '\n'+ '\t Name :' +  male_instance['name']+ '\n'+ '\t Screen Name :'+ male_instance['screen_name']+ '\n'+ '\t Description :'+ male_instance['description']+ '\n'+ '\t Gender :'+ male_instance['gender']+ '\n' + 'Example of Female class : ' + '\n'+ '\t Name :' +  female_instance['name']+ '\n'+ '\t Screen Name :'+ female_instance['screen_name']+ '\n'+ '\t Description :'+ female_instance['description']+ '\n'+ '\t Gender :'+ female_instance['gender']+'\n'
    f = open('summary.txt', 'w')
    f.write(s)
    f.close()



def main():
    users = pickle.load(open('users.pkl', 'rb'))
    friendsList = pickle.load(open('friendList.pkl', 'rb'))
    followerDetails = pickle.load(open('followerDetails.pkl', 'rb'))
    graph = pickle.load(open('graph.pkl', 'rb'))
    count_users(users,friendsList,followerDetails,graph)





if __name__ == '__main__':
    main()
