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


import random

import util
from game import Agent, Directions
from pacman import GameState
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        foodList = newFood.asList()

        minFoodDistance = min(
            (manhattanDistance(newPos, food) for food in foodList), default=float("inf")
        )

        if any(
            manhattanDistance(newPos, ghost) < 2
            for ghost in successorGameState.getGhostPositions()
        ):
            return -float("inf")

        return successorGameState.getScore() + 1 / minFoodDistance


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.maxval(gameState, 0, 0)[0]

    def minimax(self, gameState, agentIndex, depth):
        if (
            depth == self.depth * gameState.getNumAgents()
            or gameState.isLose()
            or gameState.isWin()
        ):
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            return self.maxval(gameState, 0, depth)[1]

        else:
            return self.minval(gameState, agentIndex, depth)[1]

    def maxval(self, gameState, agentIndex, depth):
        bestAction = ("max", -float("inf"))

        for action in gameState.getLegalActions(agentIndex):
            successorAction = (
                action,
                self.minimax(
                    gameState.generateSuccessor(agentIndex, action),
                    1,
                    depth + 1,
                ),
            )

            bestAction = max(bestAction, successorAction, key=lambda item: item[1])

        return bestAction

    def minval(self, gameState, agentIndex, depth):
        worstAction = ("min", float("inf"))

        for action in gameState.getLegalActions(agentIndex):
            successor = (agentIndex + 1) % gameState.getNumAgents()

            successorAction = (
                action,
                self.minimax(
                    gameState.generateSuccessor(agentIndex, action),
                    successor,
                    depth + 1,
                ),
            )

            worstAction = min(worstAction, successorAction, key=lambda item: item[1])

        return worstAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxval(gameState, 0, 0, -float("inf"), float("inf"))[0]

    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
        if (
            depth == self.depth * gameState.getNumAgents()
            or gameState.isLose()
            or gameState.isWin()
        ):
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            return self.maxval(gameState, 0, depth, alpha, beta)[1]

        else:
            return self.minval(gameState, agentIndex, depth, alpha, beta)[1]

    def maxval(self, gameState, agentIndex, depth, alpha, beta):
        bestAction = ("max", -float("inf"))

        for action in gameState.getLegalActions(agentIndex):
            successorAction = (
                action,
                self.alphabeta(
                    gameState.generateSuccessor(agentIndex, action),
                    1,
                    depth + 1,
                    alpha,
                    beta,
                ),
            )

            bestAction = max(bestAction, successorAction, key=lambda item: item[1])

            if bestAction[1] > beta:
                return bestAction

            else:
                alpha = max(alpha, bestAction[1])

        return bestAction

    def minval(self, gameState, agentIndex, depth, alpha, beta):
        worstAction = ("min", float("inf"))

        for action in gameState.getLegalActions(agentIndex):
            successor = (agentIndex + 1) % gameState.getNumAgents()

            successorAction = (
                action,
                self.alphabeta(
                    gameState.generateSuccessor(agentIndex, action),
                    successor,
                    depth + 1,
                    alpha,
                    beta,
                ),
            )

            worstAction = min(worstAction, successorAction, key=lambda item: item[1])

            if worstAction[1] < alpha:
                return worstAction

            else:
                beta = min(beta, worstAction[1])

        return worstAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        maxDepth = self.depth * gameState.getNumAgents()
        return self.expectimax(gameState, 0, "expect", maxDepth)[0]

    def expectimax(self, gameState, agentIndex, action, depth):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return action, self.evaluationFunction(gameState)

        if agentIndex == 0:
            return self.maxval(gameState, 0, action, depth)

        else:
            return self.expval(gameState, agentIndex, action, depth)

    def maxval(self, gameState, agentIndex, action, depth):
        bestAction = (max, -float("inf"))

        for legalAction in gameState.getLegalActions(agentIndex):
            successorAction = (
                legalAction
                if depth == self.depth * gameState.getNumAgents()
                else action
            )

            successorValue = self.expectimax(
                gameState.generateSuccessor(agentIndex, legalAction),
                1,
                successorAction,
                depth - 1,
            )
            bestAction = max(bestAction, successorValue, key=lambda item: item[1])

        return bestAction

    def expval(self, gameState, agentIndex, action, depth):
        legalActions = gameState.getLegalActions(agentIndex)
        avgScore = 0
        probability = 1 / len(legalActions)

        for legalAction in legalActions:
            successor = (agentIndex + 1) % gameState.getNumAgents()
            bestAction = self.expectimax(
                gameState.generateSuccessor(agentIndex, legalAction),
                successor,
                action,
                depth - 1,
            )
            avgScore += bestAction[1] * probability

        return action, avgScore


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()

    minFoodDistance = min(
        (manhattanDistance(newPos, food) for food in foodList), default=float("inf")
    )

    ghostDistances = [
        manhattanDistance(newPos, ghost)
        for ghost in currentGameState.getGhostPositions()
    ]
    ghostDistance = min(ghostDistances, default=float("inf"))
    if ghostDistance < 2:
        return -float("inf")

    foodLeft = currentGameState.getNumFood()
    capsulesLeft = len(currentGameState.getCapsules())

    foodLeftMultiplier = 100000
    capsulesLeftMultiplier = 10000
    foodDistMultiplier = 1000

    additionalFactors = 0
    if currentGameState.isLose():
        additionalFactors -= 1000000

    elif currentGameState.isWin():
        additionalFactors += 1000000

    return (
        foodLeftMultiplier / (foodLeft + 1)
        + ghostDistance
        + foodDistMultiplier / (minFoodDistance + 1)
        + capsulesLeftMultiplier / (capsulesLeft + 1)
        + additionalFactors
    )


# Abbreviation
better = betterEvaluationFunction
