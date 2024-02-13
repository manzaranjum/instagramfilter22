
chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
    if (request.from == 'true') {
        chrome.tabs.executeScript(null, { file: "./foreground.js" }, () => {
            // alert('foregournd running');
            console.log('Started Foreground Script');
        });
    }
});


