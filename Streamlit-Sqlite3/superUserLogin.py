import streamlit as st
import sqlite3 as sql

conn = sql.connect('file:auth.db?mode=ro', uri=True)
connect = conn.cursor()


def _login_user(username, password, status):
    connect.execute('SELECT * FROM users WHERE username =? AND password =? AND status =?',
                    (username, password, status))
    data = connect.fetchall()
    return data


def _list_users(connect):
    table_data = connect.execute('SELECT username,password,status,su FROM users').fetchall()
    if table_data:
        table_data2 = list(zip(*table_data))
        st.table(
            {
                'Username': table_data2[0],
                'Password': table_data2[1],
                'Status': table_data2[2],
                'Superuser?': table_data2[3],
            }
        )
    else:
        st.write('No entries in authentication database')


def _create_users(connect, init_user='', init_pass='', init_status='', init_super=False):
    user_ = st.text_input('Enter Username', value=init_user)
    pass_ = st.text_input('Enter Password (required)', value=init_pass)
    status_ = st.text_input('Enter Status', value=init_status)
    super_ = st.checkbox('Is this a superuser?', value=init_super)
    if st.button('Update Database') and user_ and pass_:
        with connect:
            connect.execute(
                'INSERT INTO USERS(username, password, su) VALUES(?,?,?,?)',
                (user_, pass_, status_, super_),
            )
            st.text('Database Updated')


def _edit_users(connect):
    userlist = [x[0] for x in connect.execute("SELECT username FROM users").fetchall()]
    userlist.insert(0, "")
    edit_user = st.selectbox("Select user", options=userlist)
    if edit_user:
        user_data = connect.execute(
            "SELECT username,password,su FROM users WHERE username =?", (edit_user,)
        ).fetchone()
        _create_users(
            connect=connect,
            init_user=user_data[0],
            init_pass=user_data[1],
            init_status=user_data[2],
            init_super=user_data[3],
        )


def _delete_users(c):
    userlist = [x[0] for x in conn.execute('SELECT username FROM users').fetchall()]
    userlist.insert(0, '')
    del_user = st.selectbox('Select user', options=userlist)
    if del_user:
        if st.button(f'Press to remove {del_user}'):
            with c:
                c.execute('DELETE FROM users WHERE username =?', (del_user,))
                st.write(f'User {del_user} deleted')


def _superuser_mode():
    with sql.connect('file:auth.db?mode=rwc', uri=True) as c:
        c.execute(
            'CREATE TABLE IF NOT EXISTS USERS (id INTEGER PRIMARY KEY, username UNIQUE ON CONFLICT REPLACE, password, status, su)'
        )
        modes = {
            'View': _list_users,
            'Create': _create_users,
            'Edit': _edit_users,
            'Delete': _delete_users,
        }
        mode = st.selectbox('Select mode', modes.keys())
        modes[mode](c)


def _user_mode():
    with sql.connect('file:auth.db?mode=rwc', uri=True) as c:
        c.execute(
            'CREATE TABLE IF NOT EXISTS USERS (id INTEGER PRIMARY KEY, username UNIQUE ON CONFLICT REPLACE, password, status, su)'
        )
        modes = {
            'View': _list_users,
        }
        mode = st.selectbox('Select mode', modes.keys())
        modes[mode](c)


if __name__ == '__main__':
    st.write(
        'Warning, superuser mode\n\n'
    )
    mainMenu = ['Login']
    choice = st.sidebar.selectbox('🔐 User Zone', mainMenu)
    username = st.sidebar.text_input('Username')
    password = st.sidebar.text_input('Password', type='password')
    status = st.sidebar.selectbox('Status', ('Super User', 'User'))

    if st.sidebar.checkbox('Login'):
        result = _login_user(username, password, status)
        if result:
            st.sidebar.success('Logged In as {}'.format(username))
            if status == 'Super User':
                _superuser_mode()
            elif status == 'User':
                _user_mode()
        else:
            st.warning('Incorrect Username/Password')
