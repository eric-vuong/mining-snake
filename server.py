#!/usr/bin/python

import os
import argparse
import cherrypy
import time

import server_logic
class BattlesnakeServer(object):
    """
    This is a simple Battlesnake server written in Python using the CherryPy Web Framework.
    For instructions see https://github.com/BattlesnakeOfficial/starter-snake-python/README.md
    """
    ready=True
    name=''
    debug=0

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def index(self):
        """
        This function is called when you register your Battlesnake on play.battlesnake.com
        See https://docs.battlesnake.com/guides/getting-started#step-4-register-your-battlesnake

        It controls your Battlesnake appearance and author permissions.
        For customization options, see https://docs.battlesnake.com/references/personalization
        
        TIP: If you open your Battlesnake URL in browser you should see this data.
        """
        print("INFO")
        return {
            "apiversion": "1",
            "author": "eric-vuong",  # TODO: Your Battlesnake Username
            "color": "#000000",  # TODO: Personalize
            "head": "default",  # TODO: Personalize
            "tail": "default",  # TODO: Personalize
            "ready": self.ready,
            "name": self.name
        }

    @cherrypy.expose
    @cherrypy.tools.json_in()
    def start(self):
        """
        This function is called everytime your snake is entered into a game.
        cherrypy.request.json contains information about the game that's about to be played.
        """
        self.ready=False
        data = cherrypy.request.json
        server_logic.open('Lucy20210812153929')
        if self.debug == 1 : print(f"{data['game']['id']} START")
        return "ok"

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def move(self):
        """
        This function is called on every turn of a game. It's how your snake decides where to move.
        Valid moves are "up", "down", "left", or "right".
        """
        data = cherrypy.request.json

        # TODO - look at the server_logic.py file to see how we decide what move to return!
        move = server_logic.choose_move(data)
        directions = ["up", "down", "left", "right"]
        move = directions[move]

        return {"move": move}
        # run post turn

    @cherrypy.expose
    @cherrypy.tools.json_in()
    def end(self):
        """
        This function is called when a game your snake was in ends.
        It's purely for informational purposes, you don't have to make any decisions here.
        """ 
        data = cherrypy.request.json
        win = -1
        #print(data)
        #print(f"{data['game']['id']} END")
        if data['board']['snakes'] != None: # it wasn't a tie
            if data['board']['snakes'][0]['id'] == data['you']['id']:
                print("we won")
                win = 1
            else:
                print("we lost")
        else:
            print("it was a tie:we lost")
        ret,loss,rewardsum = server_logic.postgame(win)
        # if the model updated, save the game lenght
        if ret != 0 : 
            f = open('./pref/'+name,'a')
            f.write(str(ret) + ","  + str(data['turn']) + ","+str(loss)+ "," + str(rewardsum) +"\n")
            f.close()
        self.ready = True
        return "ok"

    @cherrypy.expose
    def save(self):
        """
        This function will tell the snake to save a copy of itself.
        """
        path="./backup/" + self.name + time.strftime("%Y%m%d%H%M%S")
        server_logic.save(path)
        
        return "ok"

    @cherrypy.expose
    @cherrypy.tools.json_in()
    def load(self):
        data = cherrypy.request.json
        if data['path'] != None : server_logic.open(data['path'])
        else : return 400
        return 200

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--Port', default="8080", help='The port that server will be available on.')
    parser.add_argument('-n', '--Name', default="Snake", help='The name of the snake, used for logging.')
    parser.add_argument('-l', '--Load', default=None, help='Filepath to a model to load and run for this snake')
    args = vars(parser.parse_args())
    port = args["Port"]
    name = args["Name"]
    print(port)

    server = BattlesnakeServer()
    server.name = name

    cherrypy.config.update({"server.socket_host": "0.0.0.0"})
    cherrypy.config.update(
        {"server.socket_port": int(os.environ.get("PORT", port)),}
    )
    print("Starting Battlesnake Server...")
    cherrypy.quickstart(server)
