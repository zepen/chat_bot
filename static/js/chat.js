/**
 * 键盘事件
 * 按Enter键发送信息
 * **/
$(document).keydown(function(event){
    if(event.keyCode === 13){
        SendMsg();
    }
});

/**
 * 前台信息处理
 * **/
function SendMsg()
{
    let text = document.getElementById("text");
    if (text.value === "" || text.value == null)
    {
        alert("发送信息为空，请输入！")
    }
    else
    {
        AddMsg('default', SendMsgDispose(text.value));
        let retMsg = AjaxSendMsg(text.value);
        AddMsg('NLAI', retMsg);
        text.value="";
    }
}

/**
 * 发送的信息处理
 * **/
function SendMsgDispose(detail)
{
    detail = detail.replace("\n", "<br>").replace(" ", "&nbsp;");
    return detail;
}

/**
 * 增加信息
 * **/
function AddMsg(user,content)
{
    let str = CreateMsg(user, content);
    let msgs = document.getElementById("msgs");
    msgs.innerHTML = msgs.innerHTML + str;
}

/**
 * 生成回复内容
 * @return {string}
 * **/
function CreateMsg(user, content)
{
    let str = "";
    if(user === 'default')
    {
        str = "<div class=\"msg guest\"><div class=\"msg-right\"><div class=\"msg-host headDefault\" style=\"background-image: url(/static/img/user.jpeg)\"></div><div class=\"msg-ball\" title=\"今天 17:52:06\">" + content +"</div></div></div>";
    }
    else
    {
        str = "<div class=\"msg robot\"><div class=\"msg-left\" worker=\"" + user + "\"><div class=\"msg-host photo\" style=\"background-image: url(/static/img/robot.jpeg)\"></div><div class=\"msg-ball\" title=\"今天 17:52:06\">" + content + "</div></div></div>";
    }
    return str;
}

/**
 * 后台信息处理
 * @return {string}
 */
function AjaxSendMsg(_content)
{
    let retStr = "";
    console.log(_content);
    $.ajax({
        type: "POST",
        async: false,
        url: "/api",
        contentType: "application/json;charset=utf-8",
        data: '{"content": "' + _content + '"}',
        success: function (data) {
            console.log(data);
            retStr = data;
        }
    });
    return retStr;
}