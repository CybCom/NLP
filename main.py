import Test.Ackermann
import sys
sys.setrecursionlimit(999999999)
print(sys.getrecursionlimit())


def run():
    m, n = map(int, input().split())
    print(Test.Ackermann.ack(m, n))


if __name__ == '__main__':
    while True:
        run()
