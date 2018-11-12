import webapp2
import json
# from combined_competer import CombinedAutocompleter
import time
from threading import Lock

class MainPage(webapp2.RequestHandler):
    def get(self):
        self.response.out.write("""
<html>
<head>
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
<center>
<div class="ui-widget">
  <label for="animals">Type: </label>
  <textarea id="animals"></textarea>
</div>
</center>

</body>
</html>
""")

# autocompleter = CombinedAutocompleter()
# lock = Lock()
# cashed = {}


import re

class Products(webapp2.RequestHandler):
    def get(self):
        term = self.request.get('term', None)
        if '.' in term or '!' in term or '?' in term or ',' in term:
            term = re.split('\.|!|\?|,', term)[-1].lstrip()
        self.response.headers['Content-Type'] = 'application/json'
        data = [{"label":'Big cat', "value":"cat"}]
        # lock.acquire()
        # words = term.lower().split(' ')
        # part = ' '.join(words[len(words) - 5:])
        # t0 = time.time()
        # # if term in cashed.keys():
        # #     data = cashed[term]
        # # elif part in cashed:
        # #     data = cashed[part]
        # # else:
        # #     data = autocompleter.complete_combined(term.lower())
        # #     cashed[part] = data
        # data = autocompleter.complete_combined(term.lower())
        # t1 = time.time()
        # time_prediction = t1-t0
        # print term + ' -> predicted in ' + str(time_prediction)
        # lock.release()
        words = term.split(' ')
        completions = []
        for prediction in data:
            completion = {'label':prediction, 'value':prediction[len(words[-1]):]}
            completions.append(completion)
        # if term:
        #     data = [term + ' ' + i for i in data]
        completions = json.dumps(completions)
        self.response.out.write(completions)

app = webapp2.WSGIApplication([
    ('/', MainPage),
    ('/products', Products),
], debug=True)

