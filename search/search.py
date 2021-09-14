# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from util import Stack

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    fringe = Stack()
    visited = []
    fringe.push(problem.getStartState())
    visited.append(problem.getStartState())
    all_actions = []
    backtrack(problem, fringe, visited, all_actions)
    return all_actions
    
    # while not fringe.isEmpty():
    #     cur_state = fringe.pop()

    #     for node in problem.getSuccessors(cur_state):
    #         print(f"node: {node}")
    #         new_state, direction, step = node
    #         if new_state not in visited:
    #             print(f"node added to fringe, visited: {new_state}")
    #             fringe.push(new_state)
    #             all_actions.append(direction)
    #             visited.append(new_state)
    #             break
    # for i in all_actions:
    #     print(i)

    # return all_actions
    util.raiseNotDefined()

def backtrack(problem, fringe, visited, all_actions):
    if fringe.isEmpty():
        # print("entered fringe is empty?")
        # for node in problem.getSuccessor(visited[-1]):
        #     new_state, direction, step = node
        #     print(f"new state, direction: {new_state, direction}")
        #     visited.remove(visited[-1])
        #     if node not in visited:
        #         all_actions.append(direction)
        return

    cur_state = fringe.pop()
    print(cur_state)

    if problem.isGoalState(cur_state):
        print("entered goal?")
        for i in all_actions:
            print(f"'{i}'")
        return all_actions
    
    for node in problem.getSuccessors(cur_state):
        new_state, direction, step = node
        if new_state not in visited:
            print(f"node: {node}")
            fringe.push(new_state)
            all_actions.append(direction)
            visited.append(new_state)
        backtrack(problem, fringe, visited, all_actions)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
