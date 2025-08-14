-- auto-generated definition
create table reports
(
    id             varchar(255) not null
        primary key,
    created_at     datetime(6)  null,
    deleted_at     datetime(6)  null,
    updated_at     datetime(6)  null,
    content        varchar(255) null,
    session_id     varchar(255) not null,
    base_report_id varchar(255) null,
    user_id        varchar(255) not null,
    constraint FK2o32rer9hfweeylg7x8ut8rj2
        foreign key (user_id) references users (user_id),
    constraint FKstd14oimkg3gw3dqrqb0losrt
        foreign key (base_report_id) references reports (id)
);

-- auto-generated definition
create table dementia_analysis
(
    id         varchar(255) not null
        primary key,
    created_at datetime(6)  null,
    deleted_at datetime(6)  null,
    updated_at datetime(6)  null,
    risk_score float        not null,
    session_id varchar(255) not null,
    user_id    varchar(255) not null,
    constraint FK87tdtu1xmvruxg9fm6vxvmqet
        foreign key (user_id) references users (user_id)
);

-- auto-generated definition
create table emotion_analysis
(
    id         varchar(255) not null
        primary key,
    created_at datetime(6)  null,
    deleted_at datetime(6)  null,
    updated_at datetime(6)  null,
    angry      double       not null,
    bored      double       not null,
    happy      double       not null,
    sad        double       not null,
    session_id varchar(255) not null,
    surprised  double       not null,
    user_id    varchar(255) not null,
    constraint FKk70l960i73k45qa20yvarnpq5
        foreign key (user_id) references users (user_id)
);
