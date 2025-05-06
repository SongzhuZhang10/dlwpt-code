def run(app, *argv):
    argv = list(argv)
    argv.insert(0, '--num-workers=4')  # We assume you have a four-core, eight-thread CPU. Change the 4 if needed.
    log.info("Running: {}({!r}).main()".format(app, argv))

    app_cls = importstr(*app.rsplit('.', 1))  # This is a slightly cleaner call to import.
    app_cls(argv).main()

    log.info("Finished: {}.({!r}).main()".format(app, argv))

run('p2ch11.training.LunaTrainingApp', '--epochs=1')
