var checkInterval = 10000;

console.log("Script Running");
alert("Please wait while we get started with Monitoring it may take 30 seconds.");

setInterval(async () => {
    if (checkCommentsDivLength() > 10) {
        console.log("New comments detected. Running algorithm...");
        runAlgo();
    }
}, checkInterval);

async function runAlgo() {
    console.log("Running runAlgo() function...");

    var dataToSend = [];
    var comments = getAllNewCommentsInDiv();

    for (var i = 0; i < comments.length; i++) {
        var uuid = createUUID();
        comments[i].setAttribute("data-uuid", uuid);
        dataToSend.push({
            id: uuid,
            text: comments[i].innerText,
            prediction: null
        });
    }

    console.log("Sending request to predict comments...");
    var predictedComments = await sendRequest(dataToSend);
    console.log("Received predictions:", predictedComments);
    dataToSend = [];

    markNewCommentsAsChecked(comments);
    hideOffensiveComments(comments, predictedComments);

    console.log("runAlgo() function completed.");
}

function checkCommentsDivLength() {
    var commentsWrapper = document.querySelectorAll('.eo2As .EtaWk .XQXOT .Mr508 .C4VMK');
    var allComments = commentsWrapper;
    var comments = [];

    for (var i = 0; i < allComments.length; i++) {
        if (!allComments[i].getAttribute("data-uuid") && !allComments[i].getAttribute("data-ischecked")) {
            comments.push(allComments[i]);
        }
    }

    console.log("Total new comments detected:", comments.length);
    return comments.length;
}

function getAllNewCommentsInDiv() {
    var commentsWrapper = document.querySelectorAll('.eo2As .EtaWk .XQXOT .Mr508 .C4VMK');
    var allComments = commentsWrapper;
    var comments = [];

    for (var i = 0; i < allComments.length; i++) {
        if (!allComments[i].getAttribute("data-uuid") && !allComments[i].getAttribute("data-ischecked")) {
            comments.push(allComments[i]);
        }
    }

    console.log("Total new comments to process:", comments.length);
    return comments;
}

function markNewCommentsAsChecked(comments) {
    for (var i = 0; i < comments.length; i++) {
        comments[i].setAttribute("data-ischecked", "true");
    }

    console.log("Marked new comments as checked.");
}

function hideOffensiveComments(comments, predictedComments) {
    for (var i = 0; i < comments.length; i++) {
        if (comments[i].getAttribute("data-uuid") === predictedComments[i].id &&
            predictedComments[i].prediction === "Offensive") {
            comments[i].style.backgroundColor = "#a9ccbd";
            comments[i].style.color = "#a9ccbd";
            comments[i].style.border = "thin dotted red";
            console.log("Comment marked as offensive:", comments[i].innerText);
        }
    }
}

async function sendRequest(dataToSend) {
    return new Promise(async (resolve, reject) => {
        try {
            console.log("Sending POST request with data:", dataToSend);
            const rawResponse = await fetch('http://127.0.0.1:5000/comments_prediction', {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(dataToSend)
            });

            console.log("Received raw response:", rawResponse);
            var predictedComments = await rawResponse.json();
            console.log("Received predictions:", predictedComments);
            resolve(predictedComments);
        } catch (error) {
            console.error("Error while sending request:", error);
            reject(error);
        }
    });
}

function createUUID() {
    var dt = new Date().getTime();
    var uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        var r = (dt + Math.random() * 16) % 16 | 0;
        dt = Math.floor(dt / 16);
        return (c == 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
    return uuid;
}
