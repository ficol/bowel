$(function() {

    var bar = $('.bar');
    var percent = $('.percent');
    var status = $('#status');
   
    $('form').ajaxForm({
        beforeSend: function() {
            document.getElementById('download').disabled=true;
            status.html('Not uploaded');
            var percentVal = '0%';
            bar.width(percentVal);
            percent.html(percentVal);
        },
        uploadProgress: function(event, position, total, percentComplete) {
		status.html('Uploading...');
            var percentVal = percentComplete + '%';
            bar.width(percentVal);
            percent.html(percentVal);
        },
        complete: function(xhr) {
            if (xhr.status != 200) {
                status.html(xhr.responseText)
            }
            else {
                status.html('Uploaded. Analyzing...');
                $.ajax({
                    url: "/inference",
                    type: 'POST',
                    dataType: 'json', // added data type
                      success: function(res) {
                        console.log(res);
        
                        const status_code = res['result']
                        if(status_code == "200"){
                            status.html('Analysis finished');
                            document.getElementById('download').disabled=false;
                        }
                        else {
                            status.html('Analysis error: ' + res['error'])
                        }
                    },
                    error: function(res) {
                        console.log(res)
                        status.html('Analysis error: ' + res);
                    }
                });
            }
        }
    });
});
