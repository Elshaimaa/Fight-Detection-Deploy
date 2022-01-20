
document.getElementById("form").addEventListener("submit", function(e){
    e.preventDefault();
    fetch('http://localhost:8000/image_feed')
        .then(function(result){
            // console.log(result)
            return result.json()
        })
        .then(function(data){
            // console.log(data[1])
            data.toshow.map(function(item){
                // console.log(item)
                var el = document.createElement("img")
                el.src = item.url;
                var result = document.getElementById('result')
                result.parentNode.insertBefore(el,result.nextSibling)
            })
        })
})