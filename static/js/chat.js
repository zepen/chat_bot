////////////////////////////////////////////键盘事件////////////////////////////////

// 按Enter键发送信息
$(document).keydown(function(event){
    if(event.keyCode == 13){
        SendMsg();
    }
});






/////////////////////////////////////////////前台信息处理/////////////////////////////////////////////////////////
// 发送信息
function SendMsg()
{
    var text = document.getElementById("text");
    if (text.value == "" || text.value == null)
    {
        alert("发送信息为空，请输入！")
    }
    else
    {
        AddMsg('default', SendMsgDispose(text.value));
        var retMsg = AjaxSendMsg(text.value)
        AddMsg('小龙', retMsg);
        text.value = "";
    }
}
// 发送的信息处理
function SendMsgDispose(detail)
{
    detail = detail.replace("\n", "<br>").replace(" ", "&nbsp;")
    return detail;
}

// 增加信息
function AddMsg(user,content)
{
    var str = CreadMsg(user, content);
    var msgs = document.getElementById("msgs");
    msgs.innerHTML = msgs.innerHTML + str;
}

// 生成内容
function CreadMsg(user, content)
{
    var str = "";
    if(user == 'default')
    {
        str = "<div class=\"msg guest\"><div class=\"msg-right\"><div class=\"msg-host headDefault\"></div><div class=\"msg-ball\" title=\"今天 17:52:06\">" + content +"</div></div></div>"
    }
    else
    {
        str = "<div class=\"msg robot\"><div class=\"msg-left\" worker=\"" + user + "\"><div class=\"msg-host photo\" style=\"background-image: url(../Images/head.png)\"></div><div class=\"msg-ball\" title=\"今天 17:52:06\">" + content + "</div></div></div>";
    }
    return str;
}



/////////////////////////////////////////////////////////////////////// 后台信息处理 /////////////////////////////////////////////////////////////////////////////////

// 发送
function AjaxSendMsg(_content)
{
    var retStr = "";
    $.ajax({
        type: "POST",
        async:false,
        url: "/Home/ChatMethod/",
        data: {
            content: _content
        },
        error: function (request) {
            retStr = "你好";
        },
        success: function (data) {
            retStr = data.info;
        }
    });
    return retStr;
}