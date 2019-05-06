console.log("Hello")


function opencamera(){
   console.log("Open Camera JAtin")
   $.post( "/opencamera/", function( data ) {
   console.log(data)
  });
  console.log("End")
}

function generatecaption(){
  $.get("/getcaption/",function( data ){
    console.log(data)
  });
}
