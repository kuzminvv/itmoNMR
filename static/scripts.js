$(document).ready(function() {

    const glassOfWaterPreset = [10, 20, 30];

    const form = $("#sendData");
    const loading = $("#loading");
    const resultImg = $("#resultImg");
    const errorImg = $("#errorMessage");

    form.submit(function(e) {
        e.preventDefault();
        const data = $("#sendData").serialize();
        
        resultImg.addClass('is-hidden');
        errorImg.addClass('is-hidden');
        loading.removeClass('is-hidden');

        $.ajax({
            url: "/run_new",
            data: data,
            method: 'POST',
            success: function(response) {
                loading.addClass('is-hidden');
                $("#resultImg0").attr('src', `data:image/png;base64,${response.img0}`);
                $("#resultImg1").attr('src', `data:image/png;base64,${response.img1}`);
                $("#resultImg2").attr('src', `data:image/png;base64,${response.img2}`);
                resultImg.removeClass('is-hidden');

            },
            error: function(response) {
                loading.addClass('is-hidden');
                errorImg.removeClass('is-hidden');
                console.log(response);
            }
        })
    });

    $("#sumbitForm").on('click', function() { form.submit(); })

});