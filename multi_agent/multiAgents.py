# multiAgents.py
# --------------
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
Group member: 
Shuyao Tan (stan29@jhu.edu)
Yuyang Zhou (yzhou193@jhu.edu)

Some of our codes are different, due to our own preferences of code writing.
However, the ideas behind them are the same.
"""

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

from sys import maxsize
from copy import deepcopy

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # print(f"bestindex {bestIndices}")
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        import sys
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghost_point = 0
        for i in range(len(newGhostStates)):
            ghost_pos = successorGameState.getGhostPosition(i+1)
            ghost_dist = manhattanDistance(ghost_pos, newPos)
            # if ghost approaches pacman, tell pacman not to get nearer by adding this punitive item
            if ghost_dist != 0 and ghost_dist <= 4:
                ghost_point = - 1 / ghost_dist
        
        food_pos = newFood.asList()
        closest_dist = maxsize
        food_point = 0
        for dot_index in range(len(food_pos)):
            food_dist = manhattanDistance(food_pos[dot_index], newPos)
            closest_dist = min(closest_dist, food_dist)
            # as dist to food decreases, the impact of it increases. So we take the reverse to reflect this impact
            food_point = 1 / closest_dist
        return successorGameState.getScore() + ghost_point + food_point

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
          
          gameState.isWin()

          gameState.isLose()
        """
        "*** YOUR CODE HERE ***"

        def max_value(state, depth):
            if len(state.getLegalActions(0)) == 0 or depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state) 
            
            v = -maxsize - 1
            legal_moves = state.getLegalActions(0)
            for action in legal_moves:
                next_state = state.generateSuccessor(0, action)
                v = max(min_value(next_state, depth, agent_index=1), v)
            return v

        def min_value(state, depth, agent_index):
            if len(state.getLegalActions(agent_index)) == 0 or depth == self.depth or \
              state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = maxsize
            legal_moves = state.getLegalActions(agent_index)
            for action in legal_moves:
                next_state = state.generateSuccessor(agent_index, action)
                if agent_index < state.getNumAgents() - 1: # ghost's turn
                    v = min(min_value(next_state, depth, agent_index + 1), v)
                else: # back to pacman's turn
                    v = min(max_value(next_state, depth + 1), v)
            return v
        
        pacman_legal_moves = gameState.getLegalActions(0)
        max_val = -maxsize - 1
        max_action = pacman_legal_moves[0]

        for action in pacman_legal_moves:
            next_state = gameState.generateSuccessor(0, action)
            pacman_value = min_value(next_state, depth=0, agent_index=1)
            if pacman_value > max_val:
              max_val = pacman_value
              max_action = action
        return max_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        pacman_legal_moves = gameState.getLegalActions(0)
        max_action = pacman_legal_moves
        alpha = -maxsize - 1
        beta = maxsize

        def max_value(state, alpha, beta, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state) 
            
            v = -maxsize - 1

            legal_moves = state.getLegalActions(0)
            for action in legal_moves:
                next_state = state.generateSuccessor(0, action)
                v = max(min_value(next_state, alpha, beta, depth, agent_index=1), v)
                
                if v > beta:
                    return v

                alpha = max(alpha, v)
            return v

        def min_value(state, alpha, beta, depth, agent_index):
            if depth == self.depth or \
              state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = maxsize
            legal_moves = state.getLegalActions(agent_index)
            for action in legal_moves:
                next_state = state.generateSuccessor(agent_index, action)
                if agent_index < state.getNumAgents() - 1: # ghost's turn
                    v = min(min_value(next_state, alpha, beta, depth, agent_index + 1), v)
                else: # back to pacman's turn
                    v = min(max_value(next_state, alpha, beta, depth + 1), v)
                
                if v < alpha:
                    return v
                
                beta = min(beta, v)
            return v

        for action in pacman_legal_moves:
            next_state = gameState.generateSuccessor(0, action)
            pacman_value = min_value(next_state, alpha, beta, depth=0, agent_index=1)
            if pacman_value > alpha:
                alpha = pacman_value
                max_action = action
                
        return max_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        pacman_legal_moves = gameState.getLegalActions(0)
        max_action = pacman_legal_moves[0]
        max_val = - maxsize - 1

        def max_value(state, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state) 
            
            v = -maxsize - 1

            legal_moves = state.getLegalActions(0)
            for action in legal_moves:
                next_state = state.generateSuccessor(0, action)
                v = max(exp_value(next_state, depth, agent_index=1), v)
                
            return v

        def exp_value(state, depth, agent_index):
            if depth == self.depth or \
              state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            v = 0
            legal_moves = state.getLegalActions(agent_index)
            
            for action in legal_moves:
                probability = 1/len(legal_moves)
                next_state = state.generateSuccessor(agent_index, action)
                if agent_index < state.getNumAgents() - 1: # ghost's turn
                    v += probability * exp_value(next_state, depth, agent_index+1) 
                else:
                    v += probability * max_value(next_state, depth+1)
            return v

        for action in pacman_legal_moves:
            next_state = gameState.generateSuccessor(0, action)
            pacman_value = exp_value(next_state, depth=0, agent_index=1)
            if pacman_value > max_val:
                max_val = pacman_value
                max_action = action
        
        return max_action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      The evaluation function takes in the current GameStates (pacman.py) 
      and returns a number, where higher numbers are better.

      The code below extracts some useful information from the state, like the
      remaining food (new_food), Pacman's current position (new_pos), ghost's current position,
      and capsule's position.

      We evaluate based on the Manhattan distance between Pacman's position and the nearest ghost, 
      the whole distance from the nearest food to the last food on the board, and the whole distance 
      from the nearest capsule to the last capsule on the board. 
      
      More detailed explanation on the breakdown of the overall point (i.e, the value being returned) is described as below:
          1. the score of the current state
          2. ghost point: a punitive point to "scare" pacman away from the ghost. the nearer the distance between ghost and 
              pacman, the less this item will be
          3. food point: a rewarding point calculated by iterating through the whole food path beginning from the nearest 
              food possible. This item is represented by the inverse of distance between pacman and food (i.e. the nearer the 
              food, the higher the result)
          4. calsule point: similar to food point as described in point 3
          5. speedup point: we found the autograder is very slow when we just use the sum of 1 - 4. Thus, we decide to add this
            item as an indicator of how far the current state is between a final state
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    new_pos = currentGameState.getPacmanPosition()
    new_food = currentGameState.getFood()
    new_ghost_states = currentGameState.getGhostStates() # all ghost states
    capsules_list = currentGameState.getCapsules()

    food_list = new_food.asList()
    # punitive item: the nearer the distance between ghost and pacman, the less the evaluation result 
    ghost_point = 0
    min_dist = maxsize
    for ghost_state in new_ghost_states:
        ghost_pos = ghost_state.getPosition()
        min_dist = min(min_dist, manhattanDistance(ghost_pos, new_pos))

    ghost_point = - 1 / min_dist if min_dist > 1 else - maxsize - 1

    # reward item: the nearer the food, the higher the result
    food_point = get_reward_item(food_list, new_pos)
    capsule_point = get_reward_item(capsules_list, new_pos)
    # speedup item: to make decision easier for pacman (the more the food/capsule left, the worse the situation)
    speedup_item = -6 * (len(food_list) + len(capsules_list))

    return currentGameState.getScore() + food_point + ghost_point + capsule_point + speedup_item

def get_reward_item(item_list, new_pos):
    list_copy = deepcopy(item_list)
    sum_point = 0
    current_pos = new_pos
    for _ in list_copy[:]: # slice/mutate list to make sure iteration works correctly after remove elements
        closest_item = min(list_copy, key=lambda x: manhattanDistance(x, current_pos))
        sum_point += 1/(manhattanDistance(new_pos, closest_item))
        current_pos = closest_item
        list_copy.remove(closest_item)
    return sum_point

# Abbreviation
better = betterEvaluationFunction

