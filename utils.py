from string import Template


class DeltaTemplate(Template):
    delimiter = "%"


def strftimedelta(td, fmt):
    d = {"D": td.days}
    d["H"], rem = divmod(td.seconds, 3600)
    d["M"], d["S"] = divmod(rem, 60)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)


def timeit(method):
    def timed(*args, **kw):
        import datetime
        from timeit import default_timer as timer
        t_0     = timer()
        result  = method(*args, **kw)
        t_end   = timer()
        seconds = datetime.timedelta(seconds=t_end - t_0)
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            print("\n\nTimeDelta: {}".format(str(t_end - t_0)))
            print("Seconds  : {}".format(str(seconds)))
            kw['log_time'][name] = t_end - t_0
        else:
            # print('\n\nExecution Time')
            print('\tmethod: {}\t execution time: {}'.format(method.__name__, strftimedelta(seconds, "%D days %H hrs: %M mins : %S secs")), end="\r")
        return result

    return timed
