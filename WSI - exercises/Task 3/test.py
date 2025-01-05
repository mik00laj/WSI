import unittest

from main import run_game, GreedyAgent, MinMaxAgent, NinjaAgent


class TestMinMaxAgent(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMinMaxAgent, self).__init__(*args, **kwargs)

    def test_1(self):
        vector = [1, 5, 9000, 4, 3]
        first_agent, second_agent = MinMaxAgent(), GreedyAgent()
        run_game(vector, first_agent, second_agent)
        assert len(first_agent.numbers) == 3
        assert len(second_agent.numbers) == 2
        assert sum(first_agent.numbers) > 9000

    def test_2(self):
        vector = [2, 0, 3, 1]
        first_agent, second_agent = MinMaxAgent(), GreedyAgent()
        run_game(vector, first_agent, second_agent)
        assert first_agent.numbers == [2, 3]

    def test_3(self):
        vector = [0, 7, 0, 0]
        first_agent, second_agent = MinMaxAgent(), GreedyAgent()
        run_game(vector, first_agent, second_agent)
        assert first_agent.numbers == [0, 7]

    def test_4(self):
        vector = [4, -9, 1, -8, -2, 8, -7, -4, 0, -1, 7, -5, 5, -3, -10, 3, 2, 6, -6, 9]
        first_agent, second_agent = MinMaxAgent(), MinMaxAgent()
        run_game(vector, first_agent, second_agent)
        assert sum(first_agent.numbers) == -1

    def test_5(self):
        vector = [0, 4, 6, 22, 1, 5, 8, 9, 3, 10, 7, 8]
        first_agent, second_agent = MinMaxAgent(), MinMaxAgent()
        run_game(vector, first_agent, second_agent)
        assert sum(first_agent.numbers) == 58

    def test_6(self):
        vector = [-42]
        first_agent, second_agent = MinMaxAgent(), GreedyAgent()
        run_game(vector, first_agent, second_agent)
        assert first_agent.numbers == [-42]
        assert second_agent.numbers == []

    def test_7(self):
        vector = [6, 4, 8, 3, 1, 9, 6]
        first_agent, second_agent = MinMaxAgent(), GreedyAgent()
        run_game(vector, first_agent, second_agent)
        assert first_agent.numbers == [6, 9, 8, 1]
        assert second_agent.numbers == [6, 4, 3]

    def test_8(self):
        vector = [2, 6, 2, -6, 1, -2, -5, -6, 9, 7, 6, -4, 6, -2, -1, 3]
        first_agent, second_agent = MinMaxAgent(), GreedyAgent()
        run_game(vector, first_agent, second_agent)
        assert first_agent.numbers == [3, 6, -1, 6, 6, 9, -5, 1]

    def test_9(self):
        vector = [2, 7, -4, -9, -4, 3, 8, -7, -4, 9, 2, -10, 5, 4, -9, 3]
        first_agent, second_agent = MinMaxAgent(), GreedyAgent()
        run_game(vector, first_agent, second_agent)
        assert first_agent.numbers == [3, 7, -9, 5, -4, 8, -10, 9]

    def test_10(self):
        vector = [-7, -5, 6, -10, -3, -5, -3, 8, -6, -3, 9, 8, 3, -4, -2, -9, -5]
        first_agent, second_agent = MinMaxAgent(), GreedyAgent()
        run_game(vector, first_agent, second_agent)
        assert first_agent.numbers == [-5, -5, -10, -5, 8, -3, 8, -4, -9]

    def test_11(self):
        vector = [-9, -9, 4, -7, 1, -2, -8, -4, -6, -1, -6, -4, 7, 10, 8, 6, 6]
        first_agent, second_agent = NinjaAgent(), MinMaxAgent()
        run_game(vector, first_agent, second_agent)
        assert first_agent.numbers == [6, 8, 7, -6, -6, -8, -2, -7, -9]


if __name__ == '__main__':
    unittest.main()
