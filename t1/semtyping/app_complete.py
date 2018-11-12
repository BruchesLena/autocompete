import webapp2
import json
from x2.Typing.t1.combined_competer import CombinedAutocompleter
import time
from threading import Lock

class MainPage(webapp2.RequestHandler):
    def get(self):
        self.response.out.write("""
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


        <link rel="stylesheet" href="https://ajax.googleapis.com/ajax/libs/jqueryui/1.12.1/themes/smoothness/jquery-ui.css">
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
        <script src="https://ajax.googleapis.com/ajax/libs/jqueryui/1.12.1/jquery-ui.min.js"></script>


        <script>
          $( function() {
            function log( message ) {
              $( "<div>" ).text( message ).prependTo( "#log" );
              $( "#log" ).scrollTop( 0 );
            }

            function split( val ) {
              return val.split( /,\s*/ );
            }

                $( "#animals" )
              // don't navigate away from the field on tab when selecting an item
              .on( "keydown", function( event ) {
                if ( event.keyCode === $.ui.keyCode.TAB &&
                    $( this ).autocomplete( "instance" ).menu.active ) {
                  event.preventDefault();
                }
              }).autocomplete({
              source: "./products",
              minLength: 1,
                select: function( event, ui ) {
                  var terms = split( this.value );
                  // remove the current input
                  terms.pop();
                  // add the selected item
                  terms.push( ui.item.value );
                  // add placeholder to get the comma-and-space at the end
                  terms.push( "" );
                  this.value = this.value+ui.item.value;
                  return false;
                }

            });
          } );
        </script>

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
                  <li class="active"><a href="./autocomplete">Prediction</a></li>
                  <li><a href="./ner.py">NER</a></li>
                  <li><a href="./contact.html">Contact</a></li>
                </ul>
              </div><!--/.nav-collapse -->
            </div>
          </div>
        </div>





        <div class="ui-widget">
          <label for="animals" style="margin: 0px 50px 9px;"><h3>Type: </h3></label>
          <textarea id="animals" style="margin: 0px 50px 9px; width: 400px; height: 100px;"></textarea>
        </div>
        <div>
            <a href="https://docs.google.com/presentation/d/1cx6zWWYi4f2YOb_rqIj_yEfntZ4G4tDor6XBu5JLeOs/edit#slide=id.g3a6287b3ae_0_18" style="margin: 0px 50px 9px;">More information here</a>
        </div>

        <div class="examples" style="margin:-180px 650px 100px;">
        <h2>Try out these examples:</h2>
        <p>i want to eat some </p>
        <p>i want to drink some </p>
        <p>i want to drink t</p>
        <p>i like watching a </p>
        <p>i want to wear this white c</p>
        <p>i want to cook a </p>
        <p>i cook a cake for your </p>
        <p>i want to eat this chocolate </p>
        <p>i study math at the </p>
        <p>i like this modern </p>
        <p>i like this golden </p>
        <p>have you ever watched this </p>
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


autocompleter = CombinedAutocompleter()
lock = Lock()
cashed = {}


import re

class Products(webapp2.RequestHandler):
    def get(self):
        term = self.request.get('term', None)
        if '.' in term or '!' in term or '?' in term or ',' in term:
            term = re.split('\.|!|\?|,', term)[-1].lstrip()
        self.response.headers['Content-Type'] = 'application/json'
        # data = [{"label":'Big cat', "value":"cat"}]
        lock.acquire()
        words = term.lower().split(' ')
        part = ' '.join(words[len(words) - 5:])
        t0 = time.time()
        # if term in cashed.keys():
        #     data = cashed[term]
        # elif part in cashed:
        #     data = cashed[part]
        # else:
        #     data = autocompleter.complete_combined(term.lower())
        #     cashed[part] = data
        # data = autocompleter.complete_stat(term.lower())
        data = autocompleter.complete_combined(term.lower())
        t1 = time.time()
        time_prediction = t1-t0
        print term + ' -> predicted in ' + str(time_prediction)
        lock.release()
        words = term.split(' ')
        completions = []
        for prediction in data:
            completion = {'label':prediction, 'value':prediction[len(words[-1]):]}
            completions.append(completion)
        # if term:
        #     data = [term + ' ' + i for i in data]
        completions = json.dumps(completions)
        # completions = json.dumps(data)
        self.response.out.write(completions)

app = webapp2.WSGIApplication([
    ('/autocomplete', MainPage),
    ('/products', Products)
], debug=True)
from paste import cascade
from paste.fileapp import DirectoryApp
static_app = DirectoryApp("html")

# Create a cascade that looks for static files first, then tries the webapp



def main():
    from paste import httpserver
    mapp = cascade.Cascade([static_app, app])
    # httpserver.serve(mapp, host='127.0.0.1', port='8000')
    httpserver.serve(mapp, port='8000')
    #httpserver.serve(app, host="0.0.0.0", port="8080")

if __name__ == '__main__':
    main()