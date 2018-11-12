import cgi
from NER.ner_parser_new import NerParser
# ner_parser = NER_Parser()
ner_parser = NerParser()

print ("""
        <html>
        <head>

                <meta charset="utf-8">
        <title>Autocomplete</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="description" content="">
        <meta name="author" content="">

        <!-- Le styles -->
        <link href="http://mybootstrap.ru/wp-content/themes/clear-theme/styles/bootstrap.css" rel="stylesheet">
        <style>
          body {
            padding-top: 60px; /* 60px to make the container go all the way to the bottom of the topbar */
          }
        </style>
        <link href="http://mybootstrap.ru/wp-content/themes/clear-theme/styles/bootstrap-responsive.css" rel="stylesheet">

        <!-- HTML5 shim, for IE6-8 support of HTML5 elements -->
        <!--[if lt IE 9]>
          <script src="/wp-content/themes/clear-theme/js/html5shiv.js"></script>
        <![endif]-->

        <!-- Fav and touch icons -->
        <link rel="apple-touch-icon-precomposed" sizes="144x144" href="http://mybootstrap.ru/wp-content/themes/clear-theme/img/apple-touch-icon-144-precomposed.png">
        <link rel="apple-touch-icon-precomposed" sizes="114x114" href="http://mybootstrap.ru/wp-content/themes/clear-theme/img/apple-touch-icon-114-precomposed.png">
          <link rel="apple-touch-icon-precomposed" sizes="72x72" href="http://mybootstrap.ru/wp-content/themes/clear-theme/img/apple-touch-icon-72-precomposed.png">
                        <link rel="apple-touch-icon-precomposed" href="http://mybootstrap.ru/wp-content/themes/clear-theme/img/apple-touch-icon-57-precomposed.png">
                                       <link rel="shortcut icon" href="http://mybootstrap.ru/wp-content/themes/clear-theme/img/favicon.png">

        </head>


        <body>

                <div class="navbar navbar-inverse navbar-fixed-top">
          <div class="navbar-inner">
            <div class="container">
              <button type="button" class="btn btn-navbar" data-toggle="collapse" data-target=".nav-collapse">
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
              </button>
              <a class="brand" href="#">NLP</a>
              <div class="nav-collapse collapse">
                <ul class="nav">
                  <li><a href="/">Home</a></li>
                  <li><a href="./autocomplete">Prediction</a></li>
                  <li class="active"><a href="./app_ner.py">NER</a></li>
                  <li><a href="./contact.html">Contact</a></li>
                </ul>
              </div><!--/.nav-collapse -->
            </div>
          </div>
        </div>



        <div class="ui-widget">
            <form action="./app_ner.py" method="post">
          <label for="text" style="margin: 0px 50px 9px;">Text (up to 50 tokens): </label>
          <textarea name = "text" id="text" style="margin: 0px 50px 9px; width: 400px; height: 100px;"></textarea>
          <input type="submit" value="Parse" />
          </form>
        </div>
        <div>
            <p> Entities:
                <span style="color:Goldenrod;">Location</span>
                <span style="color:Brown;">Organization</span>
                <span style="color:OliveDrab;">Person</span>
            </p>
        </div>


                <!-- Le javascript
        ================================================== -->
        <!-- Placed at the end of the document so the pages load faster -->
        <script src="http://mybootstrap.ru/wp-content/themes/clear-theme/js/jquery.js"></script>
        <script src="http://mybootstrap.ru/wp-content/themes/clear-theme/js/bootstrap-transition.js"></script>
        <script src="http://mybootstrap.ru/wp-content/themes/clear-theme/js/bootstrap-alert.js"></script>
        <script src="http://mybootstrap.ru/wp-content/themes/clear-theme/js/bootstrap-modal.js"></script>
        <script src="http://mybootstrap.ru/wp-content/themes/clear-theme/js/bootstrap-dropdown.js"></script>
        <script src="http://mybootstrap.ru/wp-content/themes/clear-theme/js/bootstrap-scrollspy.js"></script>
        <script src="http://mybootstrap.ru/wp-content/themes/clear-theme/js/bootstrap-tab.js"></script>
        <script src="http://mybootstrap.ru/wp-content/themes/clear-theme/js/bootstrap-tooltip.js"></script>
        <script src="http://mybootstrap.ru/wp-content/themes/clear-theme/js/bootstrap-popover.js"></script>
        <script src="http://mybootstrap.ru/wp-content/themes/clear-theme/js/bootstrap-button.js"></script>
        <script src="http://mybootstrap.ru/wp-content/themes/clear-theme/js/bootstrap-collapse.js"></script>
        <script src="http://mybootstrap.ru/wp-content/themes/clear-theme/js/bootstrap-carousel.js"></script>
        <script src="http://mybootstrap.ru/wp-content/themes/clear-theme/js/bootstrap-typeahead.js"></script>



        </body>
        </html>
        """)

form = cgi.FieldStorage()
text = form.getvalue('text')
try:
    parsed_text = ner_parser.parse(text)
    print ('<html>')
    print ('<body style="display:inline">')
    for token in parsed_text:
        if 'Loc' in token[1]:
            print ("""<p style="color:Goldenrod;">%s</p>""") % (token[0]+' ')
        elif 'Org' in token[1]:
            print ("""<p style="color:Brown;">%s</p>""") % (token[0] + ' ')
        elif 'Person' in token[1]:
            print ("""<p style="color:OliveDrab;">%s</p>""") % (token[0] + ' ')
        else:
            print ("""<p>%s</p>""") % (token[0] + ' ')
    print ('</body>')
    print ('</html>')
except TypeError:
    pass
