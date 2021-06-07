
function clearStats(){
  $('#stats').html('<p style="text-align: center;">Twitter Analysis comparing statistics from 5 different Twitter acounts pre and during COVID-19 pandemic</p>');
}

function getLikes(){
    $.ajax({
      url: '/likes',
      success: function(data){
        console.log('Success')
        $('#stats').html(data)
    },
    error:function(errMsg){
      $('#stats').html(errMsg);
    }
  });
  var totalData = $('#p-total-likes').data();
  console.log(totalData)
  }
  
function getRetweets(){
    $.ajax({
      url: '/retweets',
      success: function(data){
        console.log('Success')
        $('#stats').html(data)
    },
    error:function(errMsg){
      $('#stats').html(errMsg);
    }
  });
  }

  function getReplies(){
    $.ajax({
      url: '/replies',
      success: function(data){
        console.log('Success')
        $('#stats').html(data)
    },
    error:function(errMsg){
      $('#stats').html(errMsg);
    }
  });
  }
  function getSentiment(){
    // delay for 5 seconds before retry
    document.getElementById("sentiment-btn").disabled = true;
    setTimeout(function(){document.getElementById("sentiment-btn").disabled = false;},5000);
    $.ajax({
      url: '/sentiment',
      success: function(data){
        console.log('Success')
        $('#stats').html(data)
    },
    error:function(errMsg){
      $('#stats').html(errMsg);
    }
  });

  }

