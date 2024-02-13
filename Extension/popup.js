document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('bt2').addEventListener('click', myFunction);

    function myFunction() {
        var vall = "";
        var ele = document.getElementsByName('age_check');

        for (var i = 0; i < ele.length; i++) {
            if (ele[i].checked) {
                vall = ele[i].value;
            }
        }

        if (vall === "T") {
            // Send a message to the background script
            chrome.runtime.sendMessage({ from: 'true', message: 'Information from popup.' });
        } else if (vall === "F") {
            alert("You are eligible for the content");
        } else {
            alert("Please Select an Option");
        }
    }
});
