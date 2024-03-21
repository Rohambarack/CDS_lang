def main():
    i = 0
    while True:
        print(i)
        i +=1
# pause and restart a process
# ctr+z pause
# fg bring back to foreground

#tmux 
# start tmux session
#  tmux new -s counter
# shortcuts: ctr+b , d   detaches from session
# - tmux ls     see sessions in background
# - tmux a -t counter    attach back to counter

if __name__ == "__main__":
    main()
