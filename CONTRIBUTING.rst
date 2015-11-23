============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/nsh87/regressors/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "feature"
is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Regressors could always use more documentation, whether as part of the
official Regressors docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/nsh87/regressors/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `regressors` for local development.

1. Fork the `regressors` repo on GitHub.
2. Clone your fork locally, then add the original repository as an upstream::

    $ git clone git@github.com:your_name_here/regressors.git
    $ cd regressors
    $ git remote add upstream https://github.com/nsh87/regressors

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ pip install virtualenv virtualenvwrapper
    $ mkvirtualenv -r requirements_dev.txt regressors
    $ python setup.py develop

4. Create a branch for local development, branching off of `dev`::

    $ git checkout -b name-of-your-bugfix-or-feature dev

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the tests, including testing other Python versions with tox::

    $ flake8 regressors tests  # Check Python syntax
    $ python setup.py test  # Run unittest tests
    $ tox  # Run unittests and check compatibility on Python 2.6, 2.7, 3.3-5

   flake8 and tox will have been installed when you created the virtualenv above.

   In order to fully support tox, you will need to have Python 2.6, 2.7, 3.3, 3.4, and 3.5 available on your system. If you're using Mac OS X you can follow this `guide <http://ishcray.com/supporting-multiple-python-versions-with-tox>`_ to cleanly install multiple Python versions.

   If you are not able to get all tox environments working, that's fine, but take heed that a pull request that has not been tested against all Python versions might be rejected if it is not compatible with a specific version. You should try your best to get the ``tox`` command working so you can verify your code and tests against multiple Python versions. You should check Travis CI in lieu once your pull request has been submitted.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

   Write sensible commit message: read `this post <http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html>`_ and `this one <http://chris.beams.io/posts/git-commit/>`_ before writing a single commit.

7. Submit a pull request through the GitHub website to merge your feature to branch `dev`. To ensure your pull request can be automatically merged, play your commits on top of the most recent `dev` branch::

    $ git fetch upstream
    $ git checkout dev
    $ git merge upstream/dev
    $ git checkout name-of-your-bugfix-or-feature
    $ git rebase dev

   This will pull the latest changes from the main repository and let you take care of resolving any merge conflicts that might arise in order for your pull request to be merged.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 2.6, 2.7, 3.3, and 3.4, and for PyPy. Check
   https://travis-ci.org/nsh87/regressors/pull_requests
   and make sure that the tests pass for all supported Python versions.

Tips
----

To run a subset of tests::

    $ python -m unittest tests.test_regressors
