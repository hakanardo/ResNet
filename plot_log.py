from matplotlib.pyplot import plot, show

times = []
accs = []
t = 0
for l in open("slog").readlines():
    i = l.find("Time cost=")
    if i > -1:
        t += float(l[i:].split("=")[1])
        continue
    i = l.find("Validation-acc=")
    if i > -1:
        accs.append(float(l[i:].split("=")[1]))
        times.append(t)

plot(times, accs)
show()