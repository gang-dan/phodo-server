package app.gangdan.phodoapi.domain.heart;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import javax.persistence.*;

@Entity
@Table(name = "heart")
@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Heart {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long heartId;

    private Long memberId;

    private Long photoGuide;

}
